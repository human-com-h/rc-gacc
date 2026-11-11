"""Run the published modules on a synthetic example with fixture weights."""

import json
from pathlib import Path

import numpy as np
import tiktoken
import torch

from rc_gacc.compress import METHODS, compress
from rc_gacc.frozen_views import patch
from rc_gacc.local_data import jsonl
from rc_gacc.stage2_models import SCGScorer
from rc_gacc.stage3_model import CleanGate


def example():
    corpus = Path(__file__).resolve().parents[1] / "examples/selector.jsonl"
    source = next(jsonl(corpus))["context"]
    statements = [
        "The museum opens on Sundays.",
        "The preceding claim applies only when the summer schedule is active.",
        "The museum cafe has a separate entrance.",
    ]
    tok = tiktoken.get_encoding("cl100k_base")
    offset, units = 0, []
    for index, text in enumerate(statements):
        offset = source.index(text, offset)
        units.append(dict(text=text, sid=index, raw_start=offset, raw_end=offset + len(text)))
        offset += len(text) + 1
    rng = np.random.default_rng(42)
    h = rng.normal(0, 0.05, (3, 384)).astype(np.float32)
    g = rng.normal(0, 0.05, 384).astype(np.float32)
    return dict(units=units, h=h, g=g, u=np.array([0.9, 0.6, 0.1]), objective_u=np.array([1.0, 0.4, 0.1]),
                cost=np.array([len(tok.encode(text + "\n")) + 2 for text in statements]),
                cov=np.maximum(h @ g, 0)[None, :], sim=h @ h.T, edges=[],
                goal=dict(source_tokens=len(tok.encode(source))), manifest=dict(coverage_solutions={"0.1": dict(selected=[0])}))


def main():
    torch.manual_seed(42)
    row = dict(record_id="museum", group_id="museum", split="dev")
    data = example()
    scorer, gate = SCGScorer().eval(), CleanGate().eval()
    calibration = dict(temperatures=[1, 1, 1], biases=[0, 0, 0],
                       tau_select=0.0, tau_bind=1.0, tau_bind_learned=None)
    results = [compress(row, data, 0.15, method, scorer, gate, calibration, patch(data))
               for method in METHODS]
    for result in results:
        print(json.dumps({k: result[k] for k in ("method", "budget", "tokens", "reader_text")}))


if __name__ == "__main__":
    main()
