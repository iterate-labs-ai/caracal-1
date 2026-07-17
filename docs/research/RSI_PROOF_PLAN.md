# Research Plan: Same-Model Recursive Self-Improvement (foundational)

## Working title
"Closing the Asymmetry Gap: Empirical Same-Model Recursive Self-Improvement
in Small Language Models with Verifiable Rewards"

*(pivot: sai de cybersec specialist, entra em RSI foundational general-purpose)*

## Motivation

Weco.ai definiu 4-level framework de RSI (L0 Delegation → L1 Net Positive → L2 Ignition
→ L3 Inflection). Papers publicados atingem no máximo L1 candidate. **Weco recusou
certificar próprio AIDE² L2**. Nenhum sistema published passa L2 credibilmente.

**Asymmetry problem**: Todos L1-candidates usam LLM externo grande no outer loop.

| Sistema | Outer (improver) | Inner (improved) | Domain |
|---|---|---|---|
| AIDE² | Claude Opus 4.7 | Gemini 3 Flash | ML eng |
| DGM | Claude 3.5 API | Coding agents | SWE |
| HGM | GPT-5-mini | Coding agents | SWE |
| SICA | Claude 3.5 API | Own code (API) | SWE |
| DARWIN | GPT-4-class | nanoGPT training | ML |

**Isso é bootstrap**, não recursão. STOP paper (2310.02304) admite explicitamente:
*"since the LM itself is not altered, this is not full recursive self-improvement."*

Gap concreto: **zero papers demonstram same-model RSI num modelo pequeno com
verifiable rewards e statistical rigor pra provar asymptotic improvement.**

## Research questions

- **RQ1** (viability): Can a small model (3B) achieve **L1 net-positive** using
  only ITSELF as outer improver, without any larger LLM?
- **RQ2** (collapse resistance): Does RLVR (verifiable reward) prevent the
  Shafayat 2505.21444 collapse mode inherent in same-model self-training?
- **RQ3** (asymptotic vs sample-efficient): Is improvement **asymptotic** (v_N
  keeps growing) or **sample-efficient only** (fast start, plateaus)? L2 gate.
- **RQ4** (asymmetry cost): How much lift is lost switching from asymmetric
  outer (larger LLM) to same-model outer? Quantifies bootstrap contribution.
- **RQ5** (generality): Does discovered improvement generalize across domain
  (math → code → reasoning) or is it task-overfit?

## Central hypothesis

> **H1**: A specialist small model (3B, Qwen2.5-3B base) equipped with
> verifiable-reward RLVR can drive its own scaffold+prompt+weight optimization
> over N ≥ 8 generations, producing gain curves whose **asymptotic component**
> (fit via ScaleRL 2510.13786 sigmoid decomposition) is significantly greater
> than v_0 with p < 0.01, WITHOUT any external LLM in the outer loop.

Testable falsification: if asymptotic gain 95% CI includes zero, or if entropy
collapse triggers (RL-PLUS 2508.00222 detector), we reject H1.

## Deliverable primário: **modelo RSI-3B**

Não é só paper — o deliverable é um **modelo público em HF** que passa L1 gate
em math + coding, produzido por same-model RSI. Nome tentativo: **`Ignite-3B`**
(referência a Weco Level 2 "Ignition" - aspiracional).

- Base: `Qwen/Qwen2.5-3B-Instruct` (ou Qwen2.5-Coder-3B-Instruct)
- Method: same-model RSI + RLVR + N=8 generations
- Release: `iterate-labs-ai/ignite-3b-v1` HF checkpoint público
- Paper: side deliverable (methodology write-up)

## Task suite v3 (locked: 5 core, math + coding + Lean tool)

**Upgrade**: OMNI-MATH substitui OlympiadBench (4× maior, cleaner). **Putnam-Bench**
entra como dual-purpose (eval + Lean RL reward source). **MathArena live +
FrontierMath** parceados como L3 gate zero-contamination. AIME vira sanity ref só.
HARP droppado (answer set indisponível).

### Core (5 tasks)

| # | Domain | Task | Role | RLVR | 3B base | Frontier '26 |
|---|---|---|---|---|---|---|
| 1 | **Code** | **LiveCodeBench monthly** (2403.07974) | L1 held-in + L3 monthly buckets | Unit test exec | ~15-25% | ~75% |
| 2 | **Code** | **BigCodeBench-Hard** (2406.15877) | L2 asymptotic, distinct dist | Unit test exec | ~20% | ~60% |
| 3 | **Math** | **OMNI-MATH** (2410.07985) | L2 gate, 4428 Olympiad problems | Rule + verifier | ~10-15% | ~60% |
| 4 | **Math+Lean** | **Putnam-Bench** (2407.11214) | L2+ dual-purpose (eval + Lean RL reward) | Lean 4 proof | <2% | ~8% formal |
| 5 | **Math L3 gate** | **MathArena live + FrontierMath** (2505.23281 UNV + 2411.04872) | L3 rolling contest + PhD ceiling | Official + expr match | <1% FrontierMath | o3~25%, GPT-5~30% |

### Optional / sanity

| # | Task | Role |
|---|---|---|
| 6 | **AIME 2024/2025** | Sanity reference only, não gate |
| 7 | **APPS** (2105.09938) 3-tier | Extra L3 code difficulty knob se compute sobrar |
| 8 | **USACO waves** (2404.10952) | Extra L3 code Bronze→Platinum |

### Droppados

- **HARP** — answer set, sem cite confiável
- **OlympiadBench** — superseded por OMNI-MATH
- **NuminaMath-Test** — training-adjacent, contamination risk

## Lean 4 como ferramenta (integrar no modelo)

Modelo `Ignite-3B` aprende a chamar Lean prover via `tool_call` durante inference
+ treinamento. Não é bench isolado — é **capability core do modelo**.

### Stack recomendado

| Componente | Escolha | Fonte |
|---|---|---|
| **Environment** | LeanDojo | arxiv 2306.15626 |
| **Interface** | `Dojo.run_tac(state, tactic_str)` JSON-RPC | LeanDojo API |
| **Training recipe** | DeepSeek-Prover-V1.5: SFT synthetic + GRPO w/ Lean verifier binary reward | arxiv 2408.08152 |
| **Fallback proof-of-scale** | Kimina-Prover 1.5B distill | arxiv 2504.11354 |
| **Runtime** | Lean 4 daemon em CPU (~2GB RAM host), modelo em GPU | Kaggle T4 x2 compatível |
| **Data** | LeanDojo Benchmark 4 (98k proofs) + mathlib4 traces | Público |

### Novo tool schema no modelo

```json
{
  "name": "lean_prove",
  "arguments": {
    "state": "<current tactic state>",
    "tactic": "<Lean 4 tactic string>"
  }
}
```

Model output: `<tool_call>...</tool_call>` → Lean daemon executa → devolve
`TacticState` ou `error` → model itera até `no goals` (proved) ou timeout.

### Por que Lean tool?

1. **Verifier gratuito**: RLVR reward direto do Lean (binary: proved / not proved)
2. **Zero LLM-judge**: Shafayat-proof, sem collapse risk
3. **Novel angle**: 3B com Lean tool-use nativo, distillado de DeepSeek-Prover-V1.5
   recipe — combina samesse-model RSI + formal verification
4. **Dual purpose**: mesmo tool serve pra Putnam-Bench eval (bench #4) E como
   verifier no RL loop
5. **Interpretability bonus**: Lean proof trace = explicable reasoning, fecha
   AIDE² "hard to understand" gap

### Descartados explícitos (com razão)

- **MLE-Bench** (2410.07095) — T4 infeasible (Pedro confirmou)
- **SWE-Bench** (any), **Multi-SWE-Bench**, **RepoBench** — repo-scale context blow T4
- **KernelBench** — precisa A100+ (kernel stable clocks)
- **PutnamBench** (2407.11214) — Lean/Isabelle toolchain pesado
- **USAMO 2025**, **CMO 2024** — proof-based, sem RLVR (precisam judge)
- **NuminaMath-Test** — training-set overlap risk
- **Aider-polyglot** — multi-lang toolchain
- **GSM8K, HumanEval, MBPP, MMLU-Pro** — saturated + contaminados
- **ARC-AGI-2** — flatline risk @ 3B (~0-3% baseline)
- **OMNI, OMNI-EPIC** — LLM-judge Shafayat collapse risk
- **BIG-Bench-Hard** — contaminado

### Justificativa suite 5-core

1. **RLVR purity**: All 5 usam exec/exact-match, no LLM-judge → Shafayat-proof
2. **L1 gate**: LCB-easy + AIME-easy + OlympiadBench-L1 + APPS-intro ~1000 easy problems, delta measurable sub-hour eval
3. **L2 asymptotic**: BigCodeBench-Hard + OlympiadBench-mid ~1k problems, variance-controlled repeats
4. **L3 acceleration**: **4 orthogonal difficulty knobs** — OlympiadBench (10 levels), APPS (3 tiers), LCB (easy/med/hard), USACO (4 divisions). Slope-of-slope possible.
5. **Held-out generalization**: AIME 2025, LCB post-cutoff months, USACO 2025 waves. Zero overlap com held-in.
6. **Kaggle T4 x2**: 3B bf16 ~6GB, headroom QLoRA r=16-32 + vLLM 2ª T4. Total ~20h aggregate.
7. **Coverage**: code + math = 2 de 3 modos RSI literature (rStar-Math + AZR + Self-Improving Transformers).

### Decontamination protocols aplicados

- **Rolling/temporal**: LiveCodeBench (monthly), AIME 2025, USACO waves, OlympiadBench (dated subset)
- **N-gram/hash decon**: LiveCodeBench, OlympiadBench, HARP
- **Contamination probes**: MATH-Perturb Hard (2502.06453) como canary opcional

## Study design

### Conditions

| Cond | Inner | Outer | RLVR | Purpose |
|---|---|---|---|---|
| **A** | Qwen2.5-3B | (none) | n/a | Static baseline |
| **B** | Qwen2.5-3B | Claude Opus 4.8 API | Yes | Asymmetric bootstrap ceiling |
| **C** | Qwen2.5-3B | Qwen2.5-3B (self) | Yes | **Main treatment: same-model RSI** |
| **D** | Qwen2.5-3B | Qwen2.5-3B (self) | No (LLM-judge only) | Ablation: RLVR necessity |
| **E** | Qwen2.5-3B frozen | Qwen2.5-3B trainable | Yes | Ablation: outer-only recursion |
| **F** | Qwen2.5-7B | Qwen2.5-7B (self) | Yes | Scale ablation (7B same-model) |

### Recursion protocol (Cond C)

At generation `k`:

1. **Inference phase**: v(k) solves training tasks across 3 domains, collect
   trajectories + verifiable reward per task
2. **Introspection phase**: Same v(k) proposes N=8 candidate mutations to:
   - System prompt / instruction template
   - Chain-of-thought scaffold
   - Data augmentation strategy (curriculum, difficulty sampling)
   - LoRA rank / lr schedule (bounded search)
3. **Evaluation phase**: Train N=8 candidate v(k+1) LoRAs (cheap LoRA-mix), eval
   each on dev split with RLVR reward
4. **Selection phase**: Top-2 by dev reward, verified on unseen validation
   slice (contamination gate: Shafayat 1.2-5% threshold)
5. **Retention**: v(k+1) = argmax(val reward). If < v(k) + ε (ε=1pp
   pre-registered), retain v(k). Log failure.
6. Log everything: mutation type, dev delta, val delta, wall-clock, tokens,
   entropy, KL(v_k+1 || v_k).

**Total generations**: N=8. Anchor v(0) = Qwen2.5-3B-Instruct base.

## Metrics + statistical machinery

### Primary L1 gate
- **Held-out reward at gen N**: R_val(v_N) - R_val(v_0), 3 domains × bootstrap 95% CI

### Primary L2 gate (asymptotic vs sample-efficient)
- **ScaleRL sigmoid decomposition** (2510.13786): fit `R(compute) = R_∞ /
  (1 + (C_50/C)^α)` per generation. L2 confirmed iff Δ(R_∞) 95% CI > 0.
- **v_N as outer improver**: run v_N in step 2 of the recursion, applied to
  fresh v_0. Compare v_1' (produced by v_N) vs v_1 (produced by v_0) on
  held-out. McNemar paired test.

### Primary L3 gate (inflection detection)
- **Bayesian model selection** (marginal likelihood): power-law `gain = a·k^(-b)`
  vs exponential `gain = c·exp(kt)`. L3 candidate iff exp wins with BF > 10.
- **Online Bayesian change-point** on `d²(cum_gain)/dt²` (Adams-MacKay 0710.3742).
  L3 candidate iff change-point posterior > 0.9 for gen ≥ 3.

### Nested-loop noise decomposition
- **Variance-components ANOVA** over replicated runs:
  `var(inner-gain) = σ²_seed + σ²_data + σ²_mutation + σ²_outer_eval`
- Minimum 3 seeds per condition to isolate σ²_seed.
- **Bregman-conforming metric** required (Heskes 2501.18581) - use RLVR reward
  directly (not accuracy which is not Bregman).

### Stopping rule (outer loop)
- **FIRMBOUND SPRT** (2501.18059) with overshoot correction (2410.16076).
  Adaptive stopping when Δ posterior stabilizes.

### Anti-hacking checks
- **AIRA_2 Hidden Consistent Evaluation** (2603.26499) - reserved held-out slice
  never seen by outer loop
- **RL-PLUS entropy collapse detector** (2508.00222) - abort if per-token
  entropy drops > 30% between generations
- **Beyond Pass@1 detector** (2508.14029) - track pass@1 AND pass@k, flag if k
  tanks while 1 grows

### Statistical power

- N=1000 held-out per domain × 3 domains = 3000 samples
- Power to detect δ=1pp @ α=0.01 (Bonferroni for 6 conds): **~82%**

## Compute budget

**Kaggle T4 x2 relay** (5 founders × 12h × 4 sessions = 240h total)

| Session | Cond covered |
|---|---|
| S01-S02 | Cond A + B baseline (2 sessions) |
| S03-S06 | Cond C gen 0-8 (4 sessions, 2 gens each) |
| S07-S08 | Cond D + E ablations |
| S09 | Cond F 7B scale |
| S10 | L2 test (v_N as outer) all conds |
| S11-S12 | Interpretability + attribution + writeup |

Total: **144h Kaggle** (dentro budget 5 founders × 30h/sem = 150h/sem).
Anthropic API budget (Cond B only): **~$400** (Opus 4.8 outer, 8 gens × 8
candidates × 5K tokens).

## Novel contributions (10)

1. **First same-3B-weights RSI** — weight recursion, not just policy-level
2. **Same-model RSI with Lean 4 tool-use** — 3B calls Lean via `tool_call`,
   distilled from DeepSeek-Prover-V1.5 recipe. First 3B RSI paper with formal
   verifier in the loop.
3. **Sigmoid-fit L2 gate** (ScaleRL 2510.13786) — separates asymptotic from
   sample-efficient. No RSI paper reports this decomposition.
4. **Nested-loop noise decomposition** — variance-components ANOVA. First formal
   treatment for RSI.
5. **Shape-agnostic L3 estimator** — Bayesian model select + change-point.
   Nobody applied change-point tooling to RSI gain curves.
6. **Direct empirical asymmetry test** — Shafayat 2505.21444 predicts collapse,
   AZR shows RLVR can break it. Test at 3B.
7. **Contraction-rate measurement** — Zenil 2601.05280 theory, first empirical.
8. **Mutation-attribution tree** — Shapley + semantic-diff clustering, closes
   AIDE² interpretability gap. Bonus: Lean proof traces are natively explicable.
9. **Failure catalog at 3B** — which mutations collapse, which extrapolate.
10. **Scale ablation 3B → 7B** — does same-model RSI improve with scale, or
    is asymmetry required at large scale?

## Related work positioning

| Paper | Contribution | Our differential |
|---|---|---|
| **STOP** (2310.02304, COLM'24) | Scaffold-only self-mod, GPT-4 | Weight-level + smaller model |
| **Gödel Agent** (2410.04444) | Runtime scaffold rewrite, single LLM | Weight recursion + 3 domains |
| **AIDE²** (2502.13138) | Asymmetric outer, ML eng | Same-model, general RLVR domains |
| **DGM** (2505.22954) | Archive-based, API models | Single trajectory, self-recursion |
| **HGM** (2510.21614, ICLR'26 oral) | CMP metric, GPT-5-mini | Sigmoid + Bayesian L2/L3 gates |
| **AZR** (2505.03335) | Same-model RLVR, general reasoning | Adds scaffold+weight recursion, not just policy |
| **R-Zero** (2508.05004) | Same base but separate weights | Single weight trajectory |
| **Self-Improving Transformers** (2502.01612) | Linear OOD, tiny model | Production 3B, multi-domain |
| **rStar-Math** (2501.04519) | MCTS+PRM, math only | No MCTS, direct scaffold+weight |
| **SEAL** (2506.10943, MIT) | Weight self-edits | Adds scaffold+prompt layer |

## Failure-mode papers to cite AND beat

| Paper | Failure documented | Our defense |
|---|---|---|
| **Shafayat 2505.21444** | Majority-vote self-reward → collapse | RLVR exogenous reward, not LLM-judge |
| **RL-PLUS 2508.00222** | Entropy collapse + support shrinkage | Entropy detector, abort mechanism |
| **Beyond Pass@1 2508.14029** | pass@1↑ / pass@k↓ tradeoff | Track both, flag divergence |
| **Zenil 2601.05280** | Entropy decay + variance amplification when exogenous signal vanishes | RLVR IS the exogenous signal, doesn't vanish |
| **Task-Centric Theory 2602.10014** | Formal conditions for sustained RSI | Test empirically if satisfied |

## RL stack (locked, Kaggle T4 x2 tested paths)

### A — Trainer: **Unsloth GRPO** (fallback: TRL 0.14+)

Único stack 3B-GRPO realista em 2× T4 (32GB total).

| Feature | Unsloth GRPO |
|---|---|
| Base | HF TRL 0.14+ w/ `GRPOTrainer` + `reward_funcs=[fn,...]` |
| VRAM cut | ~70% via 4-bit LoRA + PagedAdamW |
| Speed | ~2× throughput vs baseline TRL |
| Colocate | vLLM sleep/wake mode |
| Reward API | Any Python callable — Lean/exec/SymPy plug direto |

**Wrappers YAML permitidos**: Axolotl ou LLaMA-Factory (ambos wrap TRL).

### B — Rollout engine: **vLLM ≥0.6** (colocate/sleep mode)

- Prefix caching + chunked prefill maduros
- ~1-3k tok/s em T4 pra 3B bf16 ( throughput)
- Colocate mode: trainer dorme, rollout acorda, GPU compartilhado
- Fallback: **SGLang** RadixAttention se rollout tree-branching dominar

### C — Verifier stack

| Domínio | Ferramenta | Fonte |
|---|---|---|
| Code exec | `bwrap` + Prime Intellect `verifiers` pkg | github.com/PrimeIntellect-ai |
| Math | **Math-Verify** (HF) = latex2sympy2 + SymPy + timeout | Open-R1 default |
| Lean 4 | **Pantograph** daemon, async worker pool, 30s timeout | github.com/lenianiva/pantograph |
| Answer grader ref | OpenAI simple-evals | github.com/openai/simple-evals |

### D — Reference recipe: **DeepScaleR** (rLLM + verl + GRPO on 1.5B Qwen)

Closest hardware/model scale. Downscale batch + LoRA pra caber em T4 x2.

- Framework: rLLM (Berkeley Agentica) built on verl
- Base: DeepSeek-R1-Distill-Qwen-1.5B
- Reward: Math-Verify
- Curriculum: 8k → 24k context progressive
- Repo: `agentica-project/rllm` + `agentica-project/deepscaler`

**Cross-reference**: AZR (2505.03335) pro self-play code half.

### Infra patterns aplicar

- **Async rollout / decoupled actor-verifier**: obrigatório pra Lean (proofs 5-60s)
- **Off-policy replay** pra expensive verifiers: PPO importance ratio clip (DeepSeek-Prover-V1.5 RLPAF)
- **KV-cache reuse** via SGLang RadixAttention ou vLLM prefix cache
- **Prime Intellect `verifiers` pkg** = current standard env abstraction 2025-26
- **Ray** — overkill 2×T4, skip

### Kaggle T4 x2 layout

```
GPU 0: model actor (3B bf16 + LoRA r=32) + Unsloth PagedAdamW
GPU 1: vLLM rollout engine (colocate sleep mode)
CPU:   Lean 4 daemon (~2GB) + bwrap sandbox + Math-Verify
Disk:  KV cache prefix + replay buffer + ckpt per generation
```

### Rejeitados

NeMo-Aligner (A100+), DeepSpeed-Chat (stale 2023), TensorRT-LLM (heavy build), Ray (overkill).

## Compute + statistical machinery table

| Component | Method | Reference |
|---|---|---|
| L1 gate | Held-out reward Δ + bootstrap CI | Standard |
| L2 asymptotic vs sample-efficient | Sigmoid fit + Δasymptote CI | ScaleRL 2510.13786 |
| L3 detection | Bayesian model selection power vs exp | 2509.09677 + 0710.3742 |
| Outer-loop stopping | SPRT with overshoot correction | FIRMBOUND 2501.18059 + 2410.16076 |
| Mutation selection | UCB-style regret bounds | TextBO 2511.12063 + Evo-MAB 2205.10113 |
| Nested-loop noise | Variance-components ANOVA | Signal&Noise 2508.13144 |
| Bootstrap CI | Cheap subsampling | 2501.10289 |
| Metric choice | Bregman divergence required | Heskes 2501.18581 |
| Hidden-consistent eval | AIRA_2 protocol | 2603.26499 |
| Per-operator attribution | Gradient fingerprints + Shapley | 2604.16242 |
| Contamination guard | 1.2-5% threshold reward-hacking onset | 2505.21444 |

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| 3B too weak pra propor mutations úteis | Cond B ceiling honest failure |
| Mode collapse / reward hacking | Cond D ablation isola RLVR, RL-PLUS detector |
| Contamination train/eval | LiveCodeBench monthly rolling + AIRA_2 hidden slice |
| Kaggle 12h limit | Checkpoint per generation, 2 gens per session |
| Statistical power | Pre-registered power analysis, N=3000 held-out |
| Overfitting single domain | 3 domains (math + code + reasoning) |
| Weco 4-level classification disputed | Report multi-metric evidence, not single claim |

## Deliverables (modelo é o produto principal)

1. **🎯 `iterate-labs-ai/ignite-3b-v1`** — HF modelo público, math + coding RSI-tuned (deliverable #1)
2. **Paper** (ICLR/NeurIPS 2027): method + machinery + result table
3. **Open-source lib**: `same-model-rsi` (pip, task-agnostic harness)
4. **RSI benchmark**: L0/L1/L2/L3 gate suite reutilizável
5. **HF checkpoints intermediários**: v_0...v_N ablation series
6. **Data**: mutation logs + attribution JSONL público
7. **Ablation tables + sigmoid/power-law fit plots**

## Timeline

| Semana | Milestone |
|---|---|
| W1 | Task suite fixed, RLVR verifiers implemented, Cond A + B baseline |
| W2 | Cond C gen 0-2, stat machinery instrumented |
| W3 | Cond C gen 3-5, noise ANOVA running |
| W4 | Cond C gen 6-8, all logs frozen |
| W5 | Cond D + E ablations |
| W6 | Cond F 7B scale ablation |
| W7 | Held-out eval + L2 test (v_N as outer) |
| W8 | Interpretability + attribution tree |
| W9 | Writeup |
| W10 | External reviewer pass, submit |

**Total: 10 semanas** (~2.5 meses).

## Success criteria (pre-registered)

**Modelo `ignite-3b-v1` ships se atender**:
- LiveCodeBench held-out delta v_8 vs v_0 >= +5pp com p < 0.01
- AIME 2025 held-out delta v_8 vs v_0 >= +2pp com p < 0.05
- Zero collapse detectable (RL-PLUS entropy check pass)

**Paper claims**:
- **Primary L1**: Cond C v_8 held-out >= Cond A + 2pp with p < 0.01 → **L1 confirmed**
- **Primary L2**: Δ(sigmoid asymptote) 95% CI > 0 → **L2 candidate** (not certify)
- **Aspirational L3**: exp fit BF > 10 vs power-law → **L3 candidate**
- **Failure acceptable**: null result on L1 é publishable IF Cond B (asymmetric)
  shows lift → prova asymmetry é required, honest negative result

## Open technical questions (research agenda)

1. What's the minimal specialist model capacity for same-model RSI?
2. Does verifier hardness (RLVR vs LLM-judge) change stability curves?
3. Do mutation operators discovered by 3B generalize to 8B / 70B?
4. Can we detect L3 signal at Kaggle scale? (probably no - publish noise bound)
5. What's the phase transition (if any) between L0 and L1 for specialist models?
6. Is same-model RSI theoretically bounded by the model's own capability
   ceiling? (Zenil 2601.05280 predicts yes; test empirically)

---

## References base

Full catalog: `docs/research/RSI_2026.md`.

Cite ordering priority:
1. Weco 4-levels + first-evidence blog posts
2. STOP 2310.02304 (gap statement)
3. Gödel Agent 2410.04444 (closest same-model precedent)
4. DGM 2505.22954 + HGM 2510.21614 (L2 candidates asymmetric)
5. AIDE² 2502.13138 (asymmetric baseline)
6. AZR 2505.03335 (same-model RLVR precedent)
7. Self-Improving Transformers 2502.01612 (linearly-growing OOD)
8. rStar-Math 2501.04519 (small model self-evo)
9. SEAL 2506.10943 (weight-level self-edits)
10. Shafayat 2505.21444 (canonical failure to beat)
11. Zenil 2601.05280 (theoretical grounding)
12. Task-Centric Theory 2602.10014 (L3 predictor)
13. ScaleRL 2510.13786 (sigmoid fit methodology)
14. RE-Bench 2411.15114 (human baseline methodology)
15. FIRMBOUND 2501.18059 (SPRT stopping)
16. AIRA_2 2603.26499 (Hidden Consistent Evaluation)
17. Illusion of Diminishing Returns 2509.09677 (long-horizon L3 signal)

** (checar antes de citar)**: CyberEvolver 2605.26195 (not applicable
now anyway), Self-Reference Introspection 2607.04277, Self-Play Only Evolves
2603.02218.
