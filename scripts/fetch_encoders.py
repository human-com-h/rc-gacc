"""Fetch the pinned encoder and NLI model assets from their original provider."""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def verify(path, asset):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != asset["sha256"] or path.stat().st_size != asset["bytes"]:
        raise ValueError(f"Unexpected bytes for {asset['path']}")


def main():
    for asset in json.loads((ROOT / "configs/encoders.json").read_text(encoding="utf-8")):
        target = ROOT / asset["path"]
        if target.exists():
            verify(target, asset)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            try:
                with urlopen(asset["url"], timeout=60) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
                verify(partial, asset)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        print(f"Verified {asset['path']}")


if __name__ == "__main__":
    main()
