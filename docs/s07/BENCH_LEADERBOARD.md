# Caracal s07 Benchmark Leaderboard - Frontier ≥ 2026

Only frontier models released Jan-Jul 2026, verified em **vendor cards oficiais** (não third-party).

Filter list:
- OpenAI: **GPT-5.4, GPT-5.5, GPT-5.6, GPT-6**
- Anthropic: **Opus 4.7, 4.8, 5** · **Sonnet 5, 5.5** · **Haiku 5**
- Google: **Gemini 3.0, 3 Pro, 2.5 Deep Think**
- xAI: **Grok 4.5, 5**
- DeepSeek: **V4, V4-Pro, R2, R3**
- Moonshot: **Kimi K2.5, K3**
- Alibaba: **Qwen3.5, Qwen4, Qwen4-Max**
- Meta: **Llama 4.5, Llama 5**

## OpenAI

### GPT-5.4

| Bench | Score | Source |
|---|---|---|
| Internal Cyber Range | **73.33%** | [deploymentsafety GPT-5.5 card](https://deploymentsafety.openai.com/gpt-5-5/cybersecurity) (retro-reported) |

### GPT-5.5

| Bench | Score | Source |
|---|---|---|
| Internal Cyber Range | **93.33% (14/15)** | [deploymentsafety GPT-5.5](https://deploymentsafety.openai.com/gpt-5-5/cybersecurity) |
| Professional CTF pass@12 | **Saturated** | same |
| CVE-Bench | slightly higher than previous | same |
| VulnLMP | qualitative only | same |
| Preparedness Framework | **High** (below Critical) | [gpt-5-5 system card](https://openai.com/index/gpt-5-5-system-card/) |

### GPT-5.6 Sol / Terra / Luna (Preview)

| Bench | Score | Source |
|---|---|---|
| Internal CTF (Sol) | **96.7% saturated** | [gpt-5-6 preview](https://deploymentsafety.openai.com/gpt-5-6-preview) |
| Internal CTF (Terra) | > 5.5, < Sol | same |
| Internal CTF (Luna) | > 5.4, < 5.5 | same |
| Preparedness | **High Cybersecurity** all three | [previewing-gpt-5-6-sol](https://openai.com/index/previewing-gpt-5-6-sol/) |

### GPT-6
Sem system card publicado.

## Anthropic

### Claude Opus 4.7

| Bench | Score | Source |
|---|---|---|
| Cybench pass@30 | **~100%** (roughly similar to Opus 4.6 baseline) | [Opus 4.7 System Card](https://www-cdn.anthropic.com/037f06850df7fbe871e206dad004c3db5fd50340/Claude%20Opus%204.7%20System%20Card.pdf) |
| CyberGym pass@1 | **73.8%** (post harness update) | same |

### Claude Opus 4.8

| Bench | Score | Source |
|---|---|---|
| CyberGym full exploit | **8.8%** (250 trials) | [Opus 4.8 System Card](https://www-cdn.anthropic.com/0b4915911bb0d19eca5b5ee635c80fef830a37ea.pdf) |
| CyberGym ≥ 0.5 partial | **68.8%** | same |
| Cybench | above Sonnet 5, below Mythos 5 | same |

### Claude Sonnet 5

| Bench | Score | Source |
|---|---|---|
| CyberGym full working exploits | **0.0%** | [Sonnet 5 System Card](https://www-cdn.anthropic.com/480e0bb54327b9622282e9c39a83a4f490ed377e/Claude%20Sonnet%205%20System%20Card.pdf) |
| Cybench | slight increase over Sonnet 4.6, below Opus 4.8 | same |

### Opus 5, Sonnet 5.5, Haiku 5
Cards ainda não publicados em anthropic.com (Jul 4 2026).

## Google DeepMind

### Gemini 3 Pro (flagship Gemini 3.0)

| Bench | Score | Source |
|---|---|---|
| Hard CTF (in-house + HTB) | **11/12 = 91.7%** | [Gemini 3 Pro FSF report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) |
| Realistic cyber CCL | **Below CCL** | same |

Comparison: Gemini 2.5 = 6/12 (50%) mesma bench.

### Gemini 2.5 Deep Think

| Bench | Score | Source |
|---|---|---|
| InterCode-CTF easy / in-house medium / HTB hard | N=32-50 attempts, per-tier % não extraído | [Gemini 2.5 Deep Think card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf) |

## xAI

### Grok 4.5, Grok 5
Sem model card publicado. Latest em data.x.ai = Grok 4.1 (Nov 17 2025).

## DeepSeek

### V4
[Model card DeepSeek-V4](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) publicado Apr 27 2026 - **sem scores de cyber bench**.

### V4-Pro, R2, R3
Sem model cards primary. Zero cyber eval published em source vendor.

## Moonshot

### Kimi K2.5
[Tech report](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf) publicado - **sem cyber eval vendor-published**.

### K3
Sem card. Vaporware/rumor.

## Alibaba - Qwen3.5, Qwen4, Qwen4-Max
Sem tech report com cyber dedicated section em qwenlm.github.io/arxiv.

## Meta - Llama 4.5, Llama 5
Sem model cards em ai.meta.com. Latest Llama 4 family.

## Realidade da comparação Caracal vs frontier ≥ 2026

**Frontier ≥ 2026 usa 100% eval interna** (Cyber Range, CyberGym, VulnLMP, in-house CTF). **Nenhum publica em bench Caracal**:

| Bench Caracal | Frontier ≥ 2026 rows? |
|---|---|
| CTI-Bench (RCM/MCQ/VSP/TAA/ATE) | ❌ Zero |
| CyberMetric 500/2K/10K | ❌ Zero |
| SecQA v1/v2 | ❌ Zero |
| SecBench 44K | ❌ Zero |
| MMLU computer_security | ❌ Zero |
| SecEval | ❌ Zero |
| CyberSOCEval | ❌ Zero |
| CS-Eval | ❌ Zero |
| CyberCertBench | ❌ Zero |
| PrimeVul / DiverseVul | ❌ Zero |
| CWE prediction | ❌ Zero |

**Conclusão**: comparação Caracal 3B vs GPT-5.4/5.5/5.6 · Opus 4.7/4.8 · Sonnet 5 · Gemini 3 Pro em Kaggle T4 é **impossível via benches abertos**. Options:

1. **Caracal roda vendor evals abertos**: Cybench (agentic - precisa docker + fora Kaggle), CyberGym (não roda Kaggle), CVE-Bench (agentic).
2. **Rodar frontier em benches Caracal via API paga**: chamar GPT-5.5, Opus 4.8, Sonnet 5, Gemini 3 Pro via API em CTI-Bench + CyberMetric + SecQA + SecBench. Custa ~$300-800 estimado por bench full.
3. **Comparar só vs peer 3-8B specialists** (Foundation-Sec-8B, Sec-Gemini v1, WhiteRabbitNeo-70B) que têm rows em CTI-Bench.

**Recomendação**: **opção 2** só se Iterate Labs quiser marketing "beat GPT-5.5" - senão **opção 3** é honesta e defensível.

## Caracal target realista (opção 3)

| Bench | Peer 3-8B specialist ref | Caracal target |
|---|---|---|
| CTI-Bench RCM | Foundation-Sec-8B 72.0 | **75-82** |
| CTI-Bench MCQ | Foundation-Sec-8B 66.2 | **80-85** |
| CyberMetric-500 | s05 já 86.8 | **88-92** |
| SecQA v1 | s05 já 98.6 | **98-99** (saturated) |
| SecBench 44K | — 3B ref | **80-88** |
| MMLU-Sec | Foundation-Sec-8B 78.2 | **82-88** |
| SecEval | Foundation-Sec-8B 84.8 | **75-85** |
| CS-Eval | — 3B ref | **80-85** |
| CyberCertBench (avg) | — 3B ref | **65-80** |
| CWE prediction | Foundation-Sec-Reasoning 75.3 | **75-82** |
| PrimeVul F1 | StarCoder2-FT 18.05 | **18-25** |

**Cyber specialist ceiling** = Sec-Gemini v1 (86% CTI). Caracal com 3B mira **pareto: 3B beat 8B, close gap to Sec-Gemini a 4-11pp**.

## Sources verificadas

Todas primary vendor cards:
- [GPT-5.5 deployment safety](https://deploymentsafety.openai.com/gpt-5-5/cybersecurity)
- [GPT-5.5 system card](https://openai.com/index/gpt-5-5-system-card/)
- [GPT-5.6 preview](https://deploymentsafety.openai.com/gpt-5-6-preview)
- [Opus 4.7 System Card](https://www-cdn.anthropic.com/037f06850df7fbe871e206dad004c3db5fd50340/Claude%20Opus%204.7%20System%20Card.pdf)
- [Opus 4.8 System Card](https://www-cdn.anthropic.com/0b4915911bb0d19eca5b5ee635c80fef830a37ea.pdf)
- [Sonnet 5 System Card](https://www-cdn.anthropic.com/480e0bb54327b9622282e9c39a83a4f490ed377e/Claude%20Sonnet%205%20System%20Card.pdf)
- [Gemini 3 Pro FSF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf)
- [Gemini 2.5 Deep Think Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf)
- [DeepSeek V4 model card](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)
- [Kimi K2.5 tech report](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf)
