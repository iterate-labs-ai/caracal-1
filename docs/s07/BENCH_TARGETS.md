# Caracal s07 - Benchmark Targets

## Targets vs Caracal s05 baseline

| Metric | Caracal s05 (atual) | s07 target | Stretch | Frontier ref |
|---|---|---|---|---|
| **CTI-Bench RCM** | 42.7% | **75-82%** | 90% | Sec-Gemini 86%, Mythos 83%, GPT-5.5 81.8%, Foundation-Sec-8B 75.3% |
| **CyberMetric-500** | 86.8% | **90-92%** | 93%+ | Llama-70B 91.8%, RedSage-8B 93.8% |
| **SecQA v1+v2** | 98.6% | maintain >=98% | - | GPT-4 99.1/98 |
| **MMLU-Security** | TBD | 80%+ | 88% | RedSage-8B 88, Foundation-Sec 75 |
| **SecBench** | TBD | 70%+ | 80% | RedSage-8B 83.62 |
| **Latency T4 x2** | ~1.5s/sample | ~3s acceptable | <2s | - |
| **Model size** | 3B + 1 LoRA | 3B + 5 LoRAs merged | - | - |

## Eval methodology (R8 wrap)

- **Strict accuracy** (exact CWE match) primary metric for CTI-RCM
- **Bootstrap 95% CI** with 10K resamples (BCa method)
- **N=1000 full CTI-Bench** (not subset 150 - underpowered for 3pp detect)
- **McNemar paired test** vs s05 + Foundation-Sec-8B + CyberSecQwen-4B
- **Bonferroni correction** at alpha=0.05/3 for 3 benchmarks
- **Macro F1** for class imbalance signal (205 CWE classes)
- **Per-class breakdown** rare CWE (bottom 50% support) F1

## Comparison panel (publication)

| Model | Size | CTI-RCM | CyberMetric | SecQA | SecBench |
|---|---|---|---|---|---|
| Caracal s07 (ours) | 3B | TBD | TBD | TBD | TBD |
| Caracal s05 | 3B | 42.7% | 86.8% | 98.6% | - |
| CyberSecQwen-4B | 4B | 66.64% | - | - | - |
| Foundation-Sec-8B | 8B | 72.0% | - | - | - |
| Foundation-Sec-Reasoning | 8B | 75.3% | - | - | - |
| CyberPal 2.0-4B | 4B | ~70% | - | - | - |
| CyberPal 2.0-20B | 20B | ~75-78% | - | - | - |
| RedSage | 8B | - | 93.80% | - | 83.62 |
| Mythos | unk | 83.1% CyberGym | - | - | - |
| Sec-Gemini v1 | unk | ~86% | - | - | - |
| GPT-5.5 | unk | 81.8% CyberGym | - | - | - |
| Foundation-Sec-8B-Reasoning RL | 8B | +5.8pp gain | - | - | - |

## Bench priorities (publication-grade)

1. **CTI-Bench RCM** - primary cyber LLM bench, 2024 NeurIPS
2. **CyberMetric-500** - cyber MCQ, Llama 70B paper
3. **MMLU-Security subset** - general knowledge cyber
4. **SecBench** - 47K MCQ, multi-domain
5. **SecQA** - already saturated, keep as sanity

## Per-stage eval gates (early stop)

- L1 LoopUS: CTI-RCM should improve +5pp over s05. If not, debug LoopUS impl
- L3 Tools: BFCL-cyber subset >70%. If not, debug Hammer recipe
- L4 RLVR: CTI-RCM should improve +12pp over L3 baseline. If not, debug reward fn
- L5 RSD round 1: +3pp gain. Stop if Δ<1pp = converged
