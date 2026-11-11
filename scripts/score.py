"""Aggregate explicit evaluation judgements and group-paired uncertainty."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from rc_gacc.local_data import jsonl

KEYS = ("record_id", "group_id", "method", "ratio", "prompt", "cap")
LABELS = ("strict", "screen_positive", "source_complete", "target_value_correct",
          "substantive_error")


def conjunction(a, b):
    if a is False or b is False:
        return False
    return True if a is True and b is True else None


def cells(rows):
    seen = set()
    groups = defaultdict(list)
    for row in rows:
        if not set(KEYS + LABELS) <= set(row):
            raise ValueError("Each result row requires all documented keys and labels")
        if any(row[k] is not None and type(row[k]) is not bool for k in LABELS):
            raise ValueError("Labels must be true, false, or null")
        if row["screen_positive"] is None:
            raise ValueError("The screening flag must be known for each planned arm")
        record, group, *cell = (row[key] for key in KEYS)
        arm = (record, *cell)
        if arm in seen:
            raise ValueError(f"Duplicate arm for {record}")
        seen.add(arm)
        groups[tuple(cell)].append(row)

    result = []
    for (method, ratio, prompt, cap), arms in sorted(groups.items()):
        confirmed = sum(r["strict"] is True for r in arms)
        unresolved = sum(r["strict"] is None for r in arms)
        screen_unresolved = sum(r["strict"] is None and r["screen_positive"] for r in arms)
        scope = [conjunction(r["source_complete"], r["target_value_correct"]) for r in arms]
        result.append(dict(
            method=method, ratio=ratio, prompt=prompt, cap=cap, questions=len(arms),
            groups=len({r["group_id"] for r in arms}), strict_confirmed=confirmed,
            strict_screen_upper=confirmed + screen_unresolved,
            strict_unrestricted_upper=confirmed + unresolved,
            scope_verified=sum(x is True for x in scope), scope_unknown=scope.count(None),
            substantive_errors=sum(r["substantive_error"] is True for r in arms),
            substantive_error_unknown=sum(r["substantive_error"] is None for r in arms),
        ))
    if not result:
        raise ValueError("No planned evaluation arms")
    return result


def paired_interval(rows, method, reference, seed=42, draws=2000):
    """Average seeds within record, then pair and resample complete source groups."""
    records = defaultdict(dict)
    groups = {}
    for row in rows:
        if row["method"] not in (method, reference):
            continue
        rid, group, seed_id = row["record_id"], row["group_id"], row["seed"]
        if groups.setdefault(rid, group) != group:
            raise ValueError("A record has conflicting source-group IDs")
        key = (row["method"], seed_id)
        value = float(row["value"])
        if key in records[rid] or not np.isfinite(value):
            raise ValueError("Duplicate or nonfinite paired value")
        records[rid][key] = value
    differences = defaultdict(list)
    for rid, values in records.items():
        left = {seed_id: val for (m, seed_id), val in values.items() if m == method}
        right = {seed_id: val for (m, seed_id), val in values.items() if m == reference}
        if not left or not right:
            raise ValueError("Paired bootstrap requires both methods for each record")
        differences[groups[rid]].append(mean(left.values()) - mean(right.values()))
    if not differences or method == reference:
        raise ValueError("Paired bootstrap requires two distinct methods and source groups")
    points = np.array([mean(differences[group]) for group in sorted(differences)])
    rng = np.random.default_rng(seed)
    bootstrap = [float(rng.choice(points, size=len(points), replace=True).mean()) for _ in range(draws)]
    return dict(method=method, reference=reference, groups=len(points), seed=seed, draws=draws,
                difference=float(points.mean()), percentile_95=np.quantile(bootstrap, [0.025, 0.975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    qa = subcommands.add_parser("qa")
    qa.add_argument("input", type=Path)
    paired = subcommands.add_parser("paired")
    paired.add_argument("input", type=Path)
    paired.add_argument("--method", required=True)
    paired.add_argument("--reference", required=True)
    args = parser.parse_args()
    rows = jsonl(args.input)
    result = cells(rows) if args.mode == "qa" else paired_interval(rows, args.method, args.reference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
