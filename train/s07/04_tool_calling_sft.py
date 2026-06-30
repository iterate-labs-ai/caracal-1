"""Caracal s07.D - Tool calling LoRA (Hammer + xLAM recipe).

Vitor session 4/7. ~12h Kaggle T4 x2.

Stage: L3 (in-model tool calling).

Papers:
- Hammer arxiv 2410.04587 (LoRA r=8 alpha=16 + function masking)
- xLAM arxiv 2409.03215 (60K JSON schema dataset)
- LoopTool arxiv 2511.09148 (closed-loop error-driven)

Flow:
1. Load caracal-s07-reasoning from s07.C
2. Generate 60K cyber tool call trajectories (xLAM-style adapted to 10 cyber tools)
3. Function masking: block misleading names (Hammer technique)
4. Negative examples: 10% invalid args, 10% wrong tool choice
5. SFT LoRA r=8 alpha=16 nas attention + MLP layers
6. Eval BFCL-cyber subset (custom)
7. Save adapter: pedroafonso2/caracal-s07-tools
"""

# TODO Vitor:
# - 10 cyber tools impl (eval/s07/tools/*.py)
# - xLAM-style trajectory generator
# - Hammer function masking impl
# - SFT trainer w/ negative examples mixed
# - cyber BFCL eval suite

CYBER_TOOLS = [
    "cwe_lookup",
    "cwe_tree_path",
    "cwe_search_kw",
    "attack_lookup",
    "cve_similar",
    "cwe_micro_rubric",
    "cvss_calc",
    "verify_fit",
    "cwe_view_filter",
    "nvd_fetch",
]


def main():
    raise NotImplementedError("s07.D scaffolding - implement after s07.C ship")


if __name__ == "__main__":
    main()
