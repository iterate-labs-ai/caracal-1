"""Caracal s07.E - Minerva RLVR (GRPO + hardness-gated ACR).

Alexandre session 5/7. ~12h Kaggle T4 x2.

Stage: L4 (RL post-training).

Paper: Minerva arxiv 2602.00513 - cyber paper irmao.

Flow:
1. Merge 3 LoRAs (loopus + reasoning + tools) -> s07-base-merged
2. GRPO config T4-friendly:
   - per_device_batch=1, num_gen=4 (T4 constraint)
   - lr 1e-6, beta 0.001
   - 500 steps
3. Hardness-gated ACR (Minerva recipe):
   - if hardness > threshold, condition rollout com answer
   - sample K=4 ACR completions per buffered prompt
   - EMA teacher tau=0.7 p=0.9
4. Hierarchical CWE reward (eval/s07/hier_reward.py)
5. Periodic distillation back every I=10 steps to answer-free
6. Save final: pedroafonso2/caracal-s07-rl
"""

# TODO Alexandre:
# - merge multi-LoRA workflow (peft merge_and_unload)
# - GRPO trainer TRL + vLLM colocate 0.5 (T4 constraint)
# - hardness gating impl (Minerva 2602.00513)
# - ACR conditioning logic
# - periodic distillation back loop
# - TextCNN trace quality filter tau=0.5 (Minerva paper appendix)

GRPO_CONFIG = {
    "per_device_train_batch_size": 1,
    "num_generations": 4,  # T4 constraint
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-6,
    "beta": 0.001,
    "max_prompt_length": 512,
    "max_completion_length": 256,
    "total_training_steps": 500,
    "vllm_mode": "colocate",
    "vllm_gpu_memory_utilization": 0.5,
}


def main():
    raise NotImplementedError("s07.E scaffolding - implement after s07.D ship")


if __name__ == "__main__":
    main()
