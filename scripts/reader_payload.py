"""Build deduplicated Reader requests from local flat extracts and questions."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import tiktoken

from rc_gacc.gacc_closeout import canonical_sha, normalize_text, reader_messages, sha
from rc_gacc.local_data import jsonl

ROOT = Path(__file__).resolve().parents[1]


def build(exports, questions, config, protocol_digest):
    by_record = defaultdict(list)
    ids = set()
    for row in questions:
        if set(row) != {"record_id", "question_id", "question"} or row["question_id"] in ids:
            raise ValueError("Question rows must have three fields and unique question IDs")
        ids.add(row["question_id"])
        by_record[row["record_id"]].append(row)
    tok = tiktoken.get_encoding("cl100k_base")
    contexts, requests, arms = {}, {}, []
    prompts = config["system_prompts"]
    generation = config["generation_config"]
    cap = config["max_new_tokens"]
    for row in exports:
        text = normalize_text(row["flat_text"])
        if len(tok.encode(text, disallowed_special=())) > row["budget"]:
            raise ValueError("Reader context exceeds its budget")
        context_key = sha(text)
        contexts[context_key] = text
        for question in by_record[row["record_id"]]:
            q = question["question"].replace("\r\n", "\n").replace("\r", "\n")
            for prompt_id, prompt in prompts.items():
                messages = reader_messages(text, q)
                messages[0]["content"] = prompt
                key = canonical_sha(dict(messages=messages, generation=generation, max_new_tokens=cap))
                requests.setdefault(key, dict(request_id=key, context_key=context_key, question=q,
                                              prompt_id=prompt_id, max_new_tokens=cap,
                                              messages_sha256=canonical_sha(messages)))
                arms.append(dict(question_id=question["question_id"], method=row["method"],
                                 ratio=row["ratio"], prompt=prompt_id, request_id=key))
    if not arms or {r["question_id"] for r in arms} != ids:
        raise ValueError("Each question requires an eligible extract")
    payload = dict(shard_id="local", deployment=config["deployment"], system_prompts=prompts,
                   generation_config=generation, protocol_sha256=protocol_digest,
                   contexts=[dict(context_key=k, text=v) for k, v in contexts.items()],
                   requests=list(requests.values()))
    return payload, arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracts", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = (ROOT / "configs/reader.json").read_bytes()
    payload, arms = build(jsonl(args.extracts), jsonl(args.questions), json.loads(config),
                          hashlib.sha256(config).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, result in (("payload.json", payload), ("arms.json", arms)):
        with (args.output / name).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False)
    print(f"{len(payload['requests'])} distinct requests for {len(arms)} arms")


if __name__ == "__main__":
    main()
