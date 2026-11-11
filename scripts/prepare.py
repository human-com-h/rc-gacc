"""Create the paper's offline features for a caller-supplied local JSONL corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import tiktoken

from rc_gacc import candidates
from rc_gacc.deterministic_encoding import ColdRecordEncoder
from rc_gacc.local_data import records
from rc_gacc.nli import NLI
from rc_gacc.offline_cache import CONFIG, extract, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = list(records(args.input))
    if not corpus:
        parser.error("The input corpus is empty")
    assets = json.loads((ROOT / "configs/encoders.json").read_text(encoding="utf-8"))
    for asset in assets:
        if sha(ROOT / asset["path"]) != asset["sha256"]:
            raise ValueError(f"Encoder asset mismatch for {asset['path']}")
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    config = dict(CONFIG, encoding_policy="cold_record_ordered24_v1",
                  model_assets_sha256={a["path"]: a["sha256"] for a in assets})
    nli = NLI(ROOT)
    tok = tiktoken.get_encoding("cl100k_base")
    args.output.mkdir(parents=True, exist_ok=True)
    cache_index = args.output / "index.jsonl"
    if cache_index.exists():
        parser.error("Output already has an index")
    index = []
    for row in corpus:
        path, _ = extract(row, args.output / "records", ColdRecordEncoder(ROOT), nli, candidates, tok, config, digest)
        index.append({k: row[k] for k in ("record_id", "group_id", "split")} |
                     dict(path=path.relative_to(args.output).as_posix(), manifest_hash=sha(path / "manifest.json")))
    with cache_index.open("x", encoding="utf-8") as stream:
        for row in index:
            stream.write(json.dumps(row) + "\n")
    print(f"Prepared {len(index)} local cache records")


if __name__ == "__main__":
    main()
