"""Build canonical LiveCodeBench JSONL (arxiv 2403.07974).

Source: livecodebench/code_generation_lite on HF. Rolling monthly buckets =
built-in temporal decontamination (release_v1..v6 as of 2026).

Usage:
    python data/ignite/build_livecodebench.py --out data/ignite/lcb.jsonl --version release_v6
    huggingface-cli upload iterate-labs-ai/ignite-lcb data/ignite/lcb.jsonl --repo-type dataset

Output schema:
{id, prompt, tests, starter_code, difficulty, month, source}
"""

import argparse
import json
from pathlib import Path

JSONL_FILES = {
    "release_v1": "test.jsonl",
    "release_v2": "test2.jsonl",
    "release_v3": "test3.jsonl",
    "release_v4": "test4.jsonl",
    "release_v5": "test5.jsonl",
    "release_v6": "test6.jsonl",
}


def build(out_path: Path, version: str = "release_v6") -> int:
    from huggingface_hub import hf_hub_download

    filename = JSONL_FILES.get(version, "test6.jsonl")
    local = hf_hub_download(
        repo_id="livecodebench/code_generation_lite",
        filename=filename,
        repo_type="dataset",
    )
    with open(local) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for i, row in enumerate(rows):
            prompt = row.get("question_content") or row.get("problem") or ""
            starter = row.get("starter_code", "")
            tests_raw = row.get("public_test_cases") or row.get("private_test_cases") or "[]"
            if isinstance(tests_raw, str):
                try:
                    tests = json.loads(tests_raw)
                except json.JSONDecodeError:
                    tests = []
            else:
                tests = tests_raw
            if not prompt:
                continue
            f.write(
                json.dumps(
                    {
                        "id": row.get("question_id") or f"lcb_{i}",
                        "prompt": prompt,
                        "starter_code": starter,
                        "tests": tests,
                        "difficulty": row.get("difficulty"),
                        "month": row.get("contest_date") or row.get("release_date"),
                        "source": f"livecodebench-{version}",
                    }
                )
                + "\n"
            )
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ignite/lcb.jsonl"))
    ap.add_argument("--version", default="release_v6")
    args = ap.parse_args()
    n = build(args.out, args.version)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
