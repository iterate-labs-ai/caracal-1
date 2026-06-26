"""T10: compara Caracal Base vs Qwen base puro no probe set.

Roda probe set 2x: uma com adapter, outra sem.
Sucesso v0 = caracal_ppl < qwen_ppl em >=80% das amostras.

Usage:
    python eval/compare_baseline.py \\
        --adapter ./ckpt-out \\
        --out eval/reports/baseline-vs-caracal.json
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_probe(adapter, base, out_path):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval" / "run_probe.py"),
        "--base",
        base,
        "--out",
        str(out_path),
    ]
    if adapter:
        cmd.extend(["--adapter", adapter])
    logger.info(f"Run: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return json.loads(out_path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Run 1: Caracal (com adapter) ===")
    caracal = run_probe(args.adapter, args.base, out_dir / "_probe_caracal.json")

    logger.info("=== Run 2: Qwen base puro ===")
    qwen = run_probe(None, args.base, out_dir / "_probe_qwen.json")

    caracal_by_id = {r["id"]: r for r in caracal["results"]}
    qwen_by_id = {r["id"]: r for r in qwen["results"]}

    wins = 0
    ties = 0
    losses = 0
    per_probe = []

    for pid, cres in caracal_by_id.items():
        qres = qwen_by_id.get(pid)
        if not qres:
            continue
        cppl = cres["ppl"]
        qppl = qres["ppl"]
        if cppl < qppl * 0.98:
            outcome = "win"
            wins += 1
        elif cppl > qppl * 1.02:
            outcome = "loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1
        per_probe.append(
            {
                "id": pid,
                "caracal_ppl": cppl,
                "qwen_ppl": qppl,
                "outcome": outcome,
                "caracal_cwe_hit": cres.get("cwe_hit"),
                "qwen_cwe_hit": qres.get("cwe_hit"),
            }
        )

    total = wins + ties + losses
    win_rate = wins / total if total else 0.0

    report = {
        "adapter": args.adapter,
        "base": args.base,
        "total": total,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": win_rate,
        "caracal_mean_ppl": caracal["mean_ppl"],
        "qwen_mean_ppl": qwen["mean_ppl"],
        "caracal_cwe_hit_rate": caracal.get("cwe_hit_rate"),
        "qwen_cwe_hit_rate": qwen.get("cwe_hit_rate"),
        "v0_threshold": 0.80,
        "v0_pass": win_rate >= 0.80,
        "per_probe": per_probe,
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {args.out}")
    logger.info(f"win_rate={win_rate:.2%} | v0_pass={report['v0_pass']}")


if __name__ == "__main__":
    main()
