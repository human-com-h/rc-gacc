"""Convert a locally obtained KEEP-Bench subset into evaluation-only views."""

import argparse
import json
from pathlib import Path

from rc_gacc.local_data import jsonl

SUBSETS = ("parallel", "conditional", "imbalanced_science", "imbalanced_policy")


def evaluation_views(rows, subset):
    seen = set()
    for row in rows:
        fields = ("id", "question_group", "context", "question")
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in fields):
            raise ValueError("The input does not match the KEEP evaluation schema")
        record = f"keep/{subset}/{row['id']}"
        if record in seen:
            raise ValueError("Duplicate KEEP record")
        seen.add(record)
        selector = dict(record_id=record, group_id=f"keep/{subset}/{row['question_group']}",
                        context=row["context"], goal=row["question"], split="final",
                        access="question_conditioned")
        question = dict(record_id=record, question_id=record, question=row["question"])
        yield selector, question


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--subset", choices=SUBSETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    views = list(evaluation_views(jsonl(args.input), args.subset))
    if not views:
        parser.error("The input subset is empty")
    args.output.mkdir(parents=True, exist_ok=False)
    for index, name in enumerate(("selector.jsonl", "questions.jsonl")):
        with (args.output / name).open("x", encoding="utf-8") as stream:
            for row in views:
                stream.write(json.dumps(row[index], ensure_ascii=False) + "\n")
    print(f"Prepared {len(views)} evaluation records")


if __name__ == "__main__":
    main()
