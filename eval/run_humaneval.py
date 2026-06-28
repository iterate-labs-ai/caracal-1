"""HumanEval pass@1 - check se Caracal nao regrediu code gen vs Qwen base.

Carrega openai_humaneval (164 problems), gera solucao, executa com timeout,
checa assertions. Reporta pass@1 + breakdown por problem.

Usage:
    python eval/run_humaneval.py --adapter ./ckpt --out humaneval.json
    python eval/run_humaneval.py --no-adapter --out humaneval-base.json
"""

import argparse
import json
import logging
import multiprocessing
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._common import load_model  # noqa: E402

logger = logging.getLogger(__name__)

GEN_PREFIX = (
    "Complete the following Python function. Output ONLY the function body, "
    "no markdown, no explanation.\n\n"
)


def load_humaneval():
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    return [
        {
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        }
        for ex in ds
    ]


def generate_completion(model, tokenizer, prompt, device, max_new):
    import torch

    full = GEN_PREFIX + prompt
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return text


def _exec_worker(code, q):
    try:
        ns = {}
        exec(code, ns)
        q.put(("pass", None))
    except Exception as e:
        q.put(("fail", f"{type(e).__name__}: {e}"))


def check_solution(problem, completion, timeout=5):
    """Executa prompt+completion+test, retorna (passed, reason)."""
    full_code = problem["prompt"] + completion + "\n\n" + problem["test"]
    full_code += f"\ncheck({problem['entry_point']})\n"

    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_exec_worker, args=(full_code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "timeout"
    if q.empty():
        return False, "no result"
    status, reason = q.get()
    return status == "pass", reason


def run(model, tokenizer, device, max_new=384, limit=None):
    """Roda HumanEval pass@1. Retorna dict de metrics. Reutilizado em run_bench_all."""
    problems = load_humaneval()
    if limit:
        problems = problems[:limit]
    results, n_pass = [], 0
    for i, prob in enumerate(problems):
        completion = generate_completion(model, tokenizer, prob["prompt"], device, max_new)
        passed, reason = check_solution(prob, completion)
        n_pass += int(passed)
        results.append(
            {
                "task_id": prob["task_id"],
                "passed": passed,
                "reason": reason if not passed else None,
            }
        )
        logger.info(f"[{i + 1}/{len(problems)}] {prob['task_id']}: {'PASS' if passed else 'FAIL'}")
    return {
        "n_total": len(problems),
        "n_pass": n_pass,
        "pass_at_1": n_pass / len(problems) if problems else 0.0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (default base puro)")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--max-new", type=int, default=384, help="Max new tokens por solucao")
    parser.add_argument("--limit", type=int, default=None, help="Limit n problems (debug)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tokenizer, device = load_model(args.adapter)
    summary = run(model, tokenizer, device, args.max_new, args.limit)
    summary["adapter"] = args.adapter
    Path(args.out).write_text(json.dumps(summary, indent=2))
    logger.info(
        f"pass@1 = {summary['pass_at_1']:.3f} ({summary['n_pass']}/{summary['n_total']}) -> {args.out}"
    )


if __name__ == "__main__":
    # Evita problema de signal handling em workers
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    main()
