"""Select evidence from verified offline features using supplied weights."""

import argparse
import json
from pathlib import Path

from rc_gacc.compress import METHODS, RATIOS, compress, load_edge_head, load_scorer
from rc_gacc.frozen_views import load, patch
from rc_gacc.local_data import jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--wk8", type=Path, required=True)
    parser.add_argument("--edge-head", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--method", choices=METHODS, default="Wk8")
    parser.add_argument("--ratio", type=float, choices=RATIOS, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method != "Wk8" and (args.edge_head is None or args.calibration is None):
        parser.error("Provide --edge-head and --calibration for the chosen method")
    scorer = load_scorer(args.wk8)
    edge_head = load_edge_head(args.edge_head) if args.edge_head else None
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    directory = args.index.resolve().parent
    with args.output.open("x", encoding="utf-8") as stream:
        for row in jsonl(args.index):
            path = (directory / row["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Cache index path escapes the cache directory")
            data = load(dict(row, path=path.as_posix()), verify=True)
            result = compress(row, data, args.ratio, args.method, scorer, edge_head,
                              calibration, patch(data) if edge_head is not None else ())
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
