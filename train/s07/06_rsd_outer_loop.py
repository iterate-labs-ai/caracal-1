"""Caracal s07.F (parte 1) - RSD recursive self-distill weekly batch.

Pedro session 6/7. ~6h (parte 1 das 12h, parte 2 = DGM).

Stage: L5 (self-improvement).

Papers:
- Self-Improving Transformers arxiv 2502.01612 (OOD 10->100 digits)
- RL via Self-Distillation arxiv 2601.20802 (3x fewer attempts)
- RISE arxiv 2407.18219 (recursive introspection)

Flow:
1. Pull NVD 2025-2026 daily feed (~10K unseen CVEs/week)
2. Caracal s07-rl generates traces (CoT + tool calls + boxed CWE)
3. Dual verifier filter:
   - Rule: pred_cwe == NVD_label
   - Judge: Haiku 4.5 batch fact >=7 / read >=4
4. Promote gold traces (~3-4K/week)
5. Retrain LoRA: 30% new + 70% original (anti-collapse, Self-Improving Transformers)
6. Round 1 -> Round 2 (stop if delta < 1pp)
7. Save: pedroafonso2/caracal-s07-rsd
"""

# TODO Pedro:
# - NVD API v2 client (data/s07/nvd_feed.py)
# - dual verifier impl
# - Haiku 4.5 batch judge
# - LoRA incremental retrain
# - round stopper

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
WEEKLY_BATCH_SIZE = 10_000
GOLD_THRESHOLD = {"rule_match": True, "judge_fact": 7, "judge_read": 4}
RETRAIN_MIX = {"new_self_data": 0.30, "original": 0.70}
MAX_ROUNDS = 3
STOP_DELTA_PP = 1.0


def main():
    raise NotImplementedError("s07.F.1 scaffolding - implement after s07.E ship")


if __name__ == "__main__":
    main()
