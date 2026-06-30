# Caracal s07 - Paper References

Consolidated from 10 rounds of research = ~100 papers + 10 HF datasets.

## Core architecture papers (cited in design)

### Latent Recursion (L1)
- [LoopUS arxiv 2605.11011](https://arxiv.org/abs/2605.11011) - post-training conversion to looped, **key trick mantem s05 frozen**
- [Ouro arxiv 2510.25741](https://arxiv.org/abs/2510.25741) - 1.4-2.6B looped matches 12B
- [Retrofitted Recurrence arxiv 2511.07384](https://arxiv.org/abs/2511.07384) - LoRA recursion adapter
- [HRM arxiv 2506.21734](https://arxiv.org/abs/2506.21734) - hierarchical reasoning model
- [HRM-Text arxiv 2605.20613](https://arxiv.org/abs/2605.20613) - Sapient 1B
- [LoopFormer arxiv 2602.11451](https://arxiv.org/abs/2602.11451) - ICLR 2026 elastic depth
- [MeSH arxiv 2510.07739](https://arxiv.org/abs/2510.07739) - memory-as-state-highways
- [Coconut arxiv 2412.06769](https://arxiv.org/abs/2412.06769) - continuous latent reasoning

### Reasoning Tokens (L2)
- [DeepSeek-R1 arxiv 2501.12948](https://arxiv.org/abs/2501.12948) - `<think></think>` pure RL
- [Quiet-STaR arxiv 2403.09629](https://arxiv.org/abs/2403.09629) - latent thought tokens
- [Fast Quiet-STaR arxiv 2505.17746](https://arxiv.org/abs/2505.17746) - Qwen2.5 +5.7%
- [Process Reward Models arxiv 2504.16828](https://arxiv.org/abs/2504.16828) - `\boxed{}` 1.5B viable
- [Mixture of Tokens arxiv 2509.21482](https://arxiv.org/abs/2509.21482) - Qwen2.5-1.5B +5-35%
- [FlashThink arxiv 2505.13949](https://arxiv.org/abs/2505.13949) - early exit -77% length

### Tool Calling (L3)
- [Hammer arxiv 2410.04587](https://arxiv.org/abs/2410.04587) - LoRA r=8 alpha=16 + function masking
- [xLAM arxiv 2409.03215](https://arxiv.org/abs/2409.03215) - 1B 75.43% BFCL, JSON schema
- [Octopus v2 arxiv 2404.01744](https://arxiv.org/abs/2404.01744) - 2B matches GPT-4 on-device
- [LoopTool arxiv 2511.09148](https://arxiv.org/abs/2511.09148) - closed-loop SOTA-8B BFCL-v3
- [ToolACE arxiv 2409.00920](https://arxiv.org/abs/2409.00920) - 26K APIs dual verification
- [Toolformer arxiv 2302.04761](https://arxiv.org/abs/2302.04761) - self-supervised seminal
- [SLM Big Tasks arxiv 2504.19277](https://arxiv.org/abs/2504.19277) - 350M-1B viable

### RL / RLVR (L4)
- [Minerva arxiv 2602.00513](https://arxiv.org/abs/2602.00513) - **cyber paper irmao** RLVR
- [Foundation-Sec-Reasoning arxiv 2601.21051](https://arxiv.org/abs/2601.21051) - +5.8pp CTI-RCM
- [Pentest-R1 arxiv 2508.07382](https://arxiv.org/abs/2508.07382) - 24.2% AutoPenBench
- [REAL framework arxiv 2602.05630](https://arxiv.org/abs/2602.05630) - +6.7% 1.5B

### Self-Improvement (L5)
- [DGM arxiv 2505.22954](https://arxiv.org/abs/2505.22954) - SWE-bench 20->50%
- [Godel Agent arxiv 2410.04444](https://arxiv.org/abs/2410.04444) - self-referential framework
- [Self-Improving Transformers arxiv 2502.01612](https://arxiv.org/abs/2502.01612) - OOD 10->100 digits
- [RL via Self-Distillation arxiv 2601.20802](https://arxiv.org/abs/2601.20802) - 3x fewer attempts
- [RISE arxiv 2407.18219](https://arxiv.org/abs/2407.18219) - recursive introspection
- [SIKeD aclanthology 2025](https://aclanthology.org/2025.findings-acl.513/) - iterative distillation

## Cyber LLM peers (target comparison)

- [CyberPal.AI arxiv 2408.09304](https://arxiv.org/abs/2408.09304)
- [CyberPal 2.0 arxiv 2510.14113](https://arxiv.org/abs/2510.14113) - SecKnowledge 2.0
- [Foundation-Sec arxiv 2504.21039](https://arxiv.org/abs/2504.21039) - CPT 5.1B tokens
- [Foundation-Sec-Instruct arxiv 2508.01059](https://arxiv.org/abs/2508.01059)
- [RedSage arxiv 2601.22159](https://arxiv.org/abs/2601.22159) - **frontier-peer 8B**
- [VulnLLM-R arxiv 2512.07533](https://arxiv.org/abs/2512.07533)
- [VulReaD arxiv 2602.10787](https://arxiv.org/abs/2602.10787) - KG-guided +30% Macro F1

## CVE-CWE classification (legacy peers)

- [V2W-BERT arxiv 2102.11498](https://arxiv.org/abs/2102.11498) - 94-97% NVD hierarchical
- [ThreatZoom arxiv 2009.11501](https://arxiv.org/abs/2009.11501) - 92% fine-grain
- [RoBERTa CVE-CWE arxiv 2603.14911](https://arxiv.org/abs/2603.14911) - 125M 87.4%
- [BERT multi-objective arxiv 2503.20831](https://arxiv.org/abs/2503.20831) - 94.30%

## HF Datasets (cyber training)

- [xamxte/cve-to-cwe](https://huggingface.co/datasets/xamxte/cve-to-cwe) - 290K Claude-refined, decontam vs CTI-Bench
- [secbench-hf/SecBench](https://huggingface.co/datasets/secbench-hf/SecBench) - 47.9K MCQ+SAQ
- [AI4Sec/cti-bench](https://huggingface.co/datasets/AI4Sec/cti-bench) - 5.6K, RCM bench
- [cyber-pal-security/SecKnowledge-Eval](https://huggingface.co/datasets/cyber-pal-security/SecKnowledge-Eval) - 4.1K CC-BY-NC
- [ethanolivertroy/nist-cybersecurity-training](https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training) - **530K CC0**
- [AlicanKiraz0/All-CVE-Records-Training-Dataset](https://huggingface.co/datasets/AlicanKiraz0/All-CVE-Records-Training-Dataset) - 300K Apache
- [CIRCL/vulnerability-cwe-patch](https://huggingface.co/datasets/CIRCL/vulnerability-cwe-patch) - 49K with patches
- [stasvinokur/cve-and-cwe-dataset-1999-2025](https://huggingface.co/datasets/stasvinokur/cve-and-cwe-dataset-1999-2025) - 280K CC0
- [tuandunghcmut/combine-llm-security-benchmark](https://huggingface.co/datasets/tuandunghcmut/combine-llm-security-benchmark) - 18.1K combined
- [zeroshot/cybersecurity-corpus](https://huggingface.co/datasets/zeroshot/cybersecurity-corpus) - 1K CC0

## HF base models (considered)

- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) - **selected, continuidade s05**
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) - alternativa (CyberSecQwen base)
- [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) - alternativa (native agentic)
- [sapientinc/HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B) - alternativa (HRM arch, risky)

## Statistical methodology (R8)

- [Bootstrap CI Wolfe Cameron 2024](https://cameronrwolfe.substack.com/p/stats-llm-evals)
- [scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- McNemar paired test - scipy.stats.contingency.mcnemar
- Wilson score interval pra N<300

## Class imbalance handling

- Focal loss + EDA + oversample = +28% F1 minority
- Hierarchical CWE supervision (V2W-BERT) = parent/child loss
- MoE-GRPO class-balanced rewards (arxiv 2603.24984)

## Related but skipped

- HRM-Text-1B switch - too risky, zero cyber prior art
- SmolLM3-3B switch - perderia s05 knowledge
- Modal A100 - Pedro decision Kaggle-only
- Full mini-CPT 500M - HRM doesn't need; Caracal s05 already cyber
