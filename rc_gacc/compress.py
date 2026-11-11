"""Run the paper's learned workpoints on caller-provided offline features."""

import math

import torch

from .guard import guarded
from .learned_selector import select
from .stage2_models import SCGScorer, base_state
from .stage2_selection import utility
from .stage3_decision import decide, render, scored
from .stage3_model import CleanGate
from .stage4_decision import apply
from .workpoints import binding_after, parent_out

METHODS = ("Wk8", "Safe", "Swap", "Guarded")
RATIOS = (0.05, 0.10, 0.15, 0.20)


def load_scorer(path):
    scorer = SCGScorer()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if set(state) == set(base_state(scorer)):
        scorer.load_state_dict(state, strict=False)
    else:
        scorer.load_state_dict(state)
    return scorer.eval()


def load_edge_head(path):
    gate = CleanGate()
    gate.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return gate.eval()


def compress(row, data, ratio, method, scorer, edge_head=None, calibration=None, overlay=()):
    if method not in METHODS or ratio not in RATIOS:
        raise ValueError("Unknown method or budget ratio")
    budget = math.floor(ratio * data["goal"]["source_tokens"])
    scores, coverage = utility(scorer, data)
    indices, certificate = select(data["cost"], budget, scores, coverage, data["sim"],
                                  beta=0.15, tau=0.8, time_limit=1.0)
    if not certificate["proven_optimal"] and not indices:
        raise RuntimeError("No nonempty feasible parent incumbent")
    parent = parent_out(row, data, indices, certificate, budget)
    if method == "Wk8":
        result = parent
    else:
        if edge_head is None or calibration is None:
            raise ValueError("Relation workpoints require an edge head and calibration")
        edges = scored(edge_head, data, overlay, indices, scores, coverage, calibration)
        if method == "Safe":
            result = decide(row, data, parent, scores, coverage, edges, calibration)
        else:
            swap = apply(row, data, parent, scores, coverage, edges, calibration, "ControlledSwap")
            swap = binding_after(row, data, swap, edges, scores, coverage, calibration)
            if method == "Swap":
                result = swap
            else:
                safe = decide(row, data, parent, scores, coverage, edges, calibration)
                use_swap = guarded(safe, swap)
                result = dict(swap if use_swap else safe)
                result["guard_choice"] = "Swap" if use_swap else "Safe"
    flat = render(data, result["selected"], [], budget)
    if result["tokens"] > budget or not flat["within_budget"]:
        raise RuntimeError("Rendered context exceeds the hard budget")
    spans = [[data["units"][i]["raw_start"], data["units"][i]["raw_end"]] for i in result["selected"]]
    return dict(result, method=method, ratio=ratio, flat_text=flat["reader_text"],
                flat_tokens=flat["tokens"], source_spans=spans)
