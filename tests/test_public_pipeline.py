import json
import tempfile
import unittest
from pathlib import Path

import torch

from rc_gacc.compress import METHODS, compress
from rc_gacc.frozen_views import patch
from rc_gacc.guard import guarded
from rc_gacc.local_data import jsonl, records
from rc_gacc.stage2_models import SCGScorer
from rc_gacc.stage3_model import CleanGate
from scripts.demo import example
from scripts.import_keep_eval import evaluation_views
from scripts.reader_payload import build
from scripts.score import cells, paired_interval

ROOT = Path(__file__).resolve().parents[1]


class PublicPipelineTests(unittest.TestCase):
    def test_workpoints_bill_exact_rendered_text(self):
        torch.manual_seed(42)
        row = dict(record_id="museum", group_id="museum", split="dev")
        data = example()
        calibration = dict(temperatures=[1, 1, 1], biases=[0, 0, 0],
                           tau_select=0.0, tau_bind=1.0, tau_bind_learned=None)
        scorer, gate = SCGScorer().eval(), CleanGate().eval()
        results = {m: compress(row, data, 0.15, m, scorer, gate, calibration, patch(data)) for m in METHODS}
        self.assertTrue(set(results["Wk8"]["selected"]) <= set(results["Safe"]["selected"]))
        choice = "Swap" if guarded(results["Safe"], results["Swap"]) else "Safe"
        self.assertEqual(results["Guarded"]["selected"], results[choice]["selected"])
        for result in results.values():
            self.assertLessEqual(result["tokens"], result["budget"])
            self.assertLessEqual(result["flat_tokens"], result["budget"])

    def test_loader_rejects_gold_and_cross_split_group(self):
        row = next(jsonl(ROOT / "examples/selector.jsonl"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text(json.dumps(dict(row, gold="summer")), encoding="utf-8")
            with self.assertRaises(ValueError):
                list(records(path))
            path.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, record_id="2", split="final")),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                list(records(path))

    def test_payload_keeps_prompt_and_method_multiplicity(self):
        config = json.loads((ROOT / "configs/reader.json").read_text(encoding="utf-8"))
        export = dict(record_id="museum", method="Wk8", ratio=0.2, budget=10, flat_text="Only in summer.")
        payload, arms = build([export, dict(export, method="Safe")],
                              jsonl(ROOT / "examples/questions.jsonl"), config, "fixture")
        self.assertEqual(len(arms), 4)
        self.assertEqual(len(payload["requests"]), 2)

    def test_keep_import_creates_final_selector_view_without_gold(self):
        fixture = dict(id="one", question_group="museum", context="An original synthetic passage.",
                       question="What is described?", answer="A passage.", key="private label")
        (selector, question), = evaluation_views([fixture], "conditional")
        self.assertEqual(selector["split"], "final")
        self.assertEqual(selector["access"], "question_conditioned")
        self.assertNotIn("key", selector)
        self.assertNotIn("answer", question)

    def test_unknown_retained_in_unrestricted_bound(self):
        row, = cells(jsonl(ROOT / "examples/qa.jsonl"))
        self.assertEqual((row["strict_confirmed"], row["strict_screen_upper"],
                          row["strict_unrestricted_upper"]), (1, 1, 2))

    def test_paired_bootstrap_resamples_source_groups(self):
        rows = [dict(record_id=r, group_id=g, method=m, seed=42, value=v)
                for r, g, m, v in [("a", "x", "Safe", 1), ("a", "x", "Wk8", 0),
                                    ("b", "x", "Safe", 1), ("b", "x", "Wk8", 0),
                                    ("c", "y", "Safe", 0), ("c", "y", "Wk8", 0)]]
        result = paired_interval(rows, "Safe", "Wk8")
        self.assertEqual(result["groups"], 2)
        self.assertAlmostEqual(result["difference"], 0.5)
