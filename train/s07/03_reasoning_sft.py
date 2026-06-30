"""Caracal s07.C - Reasoning tokens SFT (Fast Quiet-STaR + DeepSeek-R1 style).

Arthur session 3/7. ~12h Kaggle T4 x2.

Stage: L2 (reasoning tokens).

Papers:
- DeepSeek-R1 arxiv 2501.12948 (<think></think> via RL)
- Fast Quiet-STaR arxiv 2505.17746 (curriculum)
- Process Reward arxiv 2504.16828 (\\boxed{} final)

Flow:
1. Load caracal-s07-loopus from s07.B
2. Generate 30K trajectories via Caracal s05 + Sonnet 4.6 batch:
   Input: CVE -> Output: <think>...</think> ... \\boxed{CWE-NNN}
3. SFT LoRA r=32 attention layers
4. Curriculum: start short traces (4 epochs), ramp up long traces (4 epochs)
5. Save adapter: pedroafonso2/caracal-s07-reasoning
"""

# TODO Arthur:
# - Sonnet 4.6 batch API integration (eval/data/sonnet_cot_batch.py)
# - prompt template (R6.4 design)
# - SFT trainer Unsloth or transformers
# - curriculum learning controller
# - eval CTI-RCM 150 samples + bootstrap CI

TEACHER_MODEL = "claude-sonnet-4-6@default"
TEACHER_BATCH_SIZE = 100_000  # per batch (Anthropic limit)
NUM_TRAJECTORIES = 30_000
CURRICULUM_EPOCHS = [4, 4]  # short -> long


def main():
    raise NotImplementedError("s07.C scaffolding - implement after s07.B ship")


if __name__ == "__main__":
    main()
