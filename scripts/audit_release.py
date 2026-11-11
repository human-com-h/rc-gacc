"""Check the staged or checked-out release tree for out-of-scope content."""

import re
import subprocess
from pathlib import PurePosixPath

TOP = {".gitignore", ".gitattributes", "README.md", "LICENSE", "CITATION.cff",
       "CHANGELOG.md", "requirements.txt", "requirements-reader.txt"}
CONFIGS = {"configs/encoders.json", "configs/reader.json", "configs/source-provenance.json"}
EXAMPLES = {"examples/selector.jsonl", "examples/questions.jsonl", "examples/qa.jsonl"}
WORKFLOWS = {".github/workflows/ci.yml", ".github/dependabot.yml"}
SECRET = re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|\b(?:gh[pousr]_"
                    r"|github_pat_|hf_)[A-Za-z0-9_]{25,}\b|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
                    r"|[A-Za-z]:[\\/](?:Users|CP-GACC)[\\/]", re.IGNORECASE)


def allowed(name):
    if name in TOP | CONFIGS | EXAMPLES | WORKFLOWS:
        return True
    path = PurePosixPath(name)
    return len(path.parts) == 2 and (
        (path.parts[0] in {"rc_gacc", "scripts", "tests"} and path.suffix == ".py")
        or (path.parts[0] == "docs" and path.name in {"evaluation.md", "maintenance.md"})
    )


def verify(name, content):
    if not allowed(name) or b"\0" in content or len(content) > 200_000:
        raise ValueError(f"File outside release scope: {name}")
    text = content.decode("utf-8-sig")
    if SECRET.search(text) or re.search(r"(?m)^\\documentclass\b", text):
        raise ValueError(f"Sensitive content marker: {name}")


def main():
    files = subprocess.check_output(["git", "ls-files", "-s", "-z"]).decode("utf-8").split("\0")
    count = 0
    for row in filter(None, files):
        fields, name = row.split("\t", 1)
        mode, object_id, stage = fields.split()
        if mode not in ("100644", "100755") or stage != "0":
            raise ValueError(f"Unexpected Git entry: {name}")
        verify(name, subprocess.check_output(["git", "cat-file", "blob", object_id]))
        count += 1
    if not count:
        raise ValueError("The release index is empty")
    print(f"Audited {count} tracked files")


if __name__ == "__main__":
    main()
