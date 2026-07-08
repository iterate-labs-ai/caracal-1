"""Anthropic Batch API teacher generation.

Sonnet 4.6 gera CoT traces em xamxte/cve-to-cwe. ~$158 pra 234K samples.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python data/s07/caracal_s05_teacher_gen.py --n-samples 5000 --out traces.jsonl
"""

import argparse
import json
import time
from pathlib import Path

MODEL = "claude-sonnet-4-6@default"
BATCH_LIMIT = 100_000

PROMPT_TEMPLATE = """Analyze this CVE and identify the underlying CWE.
Reasoning must be inside <think>...</think>. Final answer inside \\boxed{{CWE-NNN}}.

CVE Description: {cve_description}

Analysis:"""


def build_requests(rows: list[dict]) -> list[dict]:
    return [
        {
            "custom_id": f"cve-{r['cve_id']}",
            "params": {
                "model": MODEL,
                "max_tokens": 512,
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(cve_description=r["description"]),
                    }
                ],
            },
        }
        for r in rows
    ]


def run_batch(client, requests: list[dict]) -> str:
    batch = client.messages.batches.create(requests=requests)
    print(f"batch {batch.id} created, {len(requests)} reqs")
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            return batch.id
        time.sleep(30)


def collect_results(client, batch_id: str, rows: list[dict]) -> list[dict]:
    row_by_id = {f"cve-{r['cve_id']}": r for r in rows}
    traces = []
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            continue
        cid = result.custom_id
        text = result.result.message.content[0].text
        row = row_by_id.get(cid)
        if not row:
            continue
        traces.append(
            {
                "cve_id": row["cve_id"],
                "description": row["description"],
                "cot_and_answer": text,
                "cwe_gold": row["cwe_id"],
                "teacher": MODEL,
            }
        )
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("data/caracal-s05-cot-traces.jsonl"))
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from anthropic import Anthropic
    from datasets import load_dataset

    ds = load_dataset("xamxte/cve-to-cwe", split=args.split).select(range(args.n_samples))
    rows = [
        {"cve_id": r["cve_id"], "description": r["description"], "cwe_id": r["cwe_id"]} for r in ds
    ]

    client = Anthropic()
    requests = build_requests(rows)
    for chunk_start in range(0, len(requests), BATCH_LIMIT):
        chunk = requests[chunk_start : chunk_start + BATCH_LIMIT]
        batch_id = run_batch(client, chunk)
        traces = collect_results(client, batch_id, rows[chunk_start : chunk_start + BATCH_LIMIT])
        with args.out.open("a") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        print(f"chunk {chunk_start}: {len(traces)} traces -> {args.out}")


if __name__ == "__main__":
    main()
