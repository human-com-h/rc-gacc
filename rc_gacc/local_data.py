"""Load a local, authorized selector corpus without answers or annotations."""

import json

from .registry import digest_text

FIELDS = {"record_id", "group_id", "context", "goal", "split", "access"}
TRACKS = {"goal_only": "goal_only_gacc", "question_conditioned": "question_conditioned_keep"}


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def records(path):
    seen, splits = set(), {}
    for row in jsonl(path):
        if set(row) != FIELDS or not all(isinstance(x, str) and x.strip() for x in row.values()):
            raise ValueError("Selector JSONL must contain exactly the six documented text fields")
        record_id, group_id, split = (row[key] for key in ("record_id", "group_id", "split"))
        if record_id in seen or splits.setdefault(group_id, split) != split:
            raise ValueError("Duplicate record or source group crossing splits")
        seen.add(record_id)
        if row["access"] not in TRACKS or split not in {"train", "dev", "calibration", "final"}:
            raise ValueError("Unknown information-access track or split")
        yield dict(record_id=record_id, group_id=group_id, split=split, context_id=group_id,
                   context=row["context"], goal=row["goal"], track=TRACKS[row["access"]],
                   source_sha256=digest_text(row["context"]))
