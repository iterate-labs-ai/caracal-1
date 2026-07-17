# Ignite-3B RSI Strategy · How to beat Weco AIDE²

Consolidação final: KV cache combos + agentic patterns + reasoning infra + Weco gaps + `weco` CLI baseline. Estratégia direta pra shipp modelo + paper que supere first-evidence claim da Weco.

Venue target: **ICLR 2026 RSI Workshop** (openreview OsPQ6zTQXV) + arxiv + blog.

---

## 1. O que Weco realmente entregou (AIDE² first-evidence)

| Item | AIDE² status |
|---|---|
| Paper arxiv | **2502.13138 é só AIDE original** (inner loop). AIDE² technical report ainda "forthcoming". |
| Blog | https://weco.ai/blog/first-evidence-of-recursive-self-improvement — 3-5K words |
| Numeric result | AIDE47 +0.053 (p=0.0024), AIDE85 +0.042 (p=0.0041) em MLE-Bench Lite |
| Reward-hacking | 63% → 34% via 3-layer defense (não zerou) |
| Level claim | **Level 2 net-positive delegation** — NÃO L3 ignition |
| Compute | 8 dias unattended, 100 outer-loop steps |
| Repro | ❌ full code + ckpt + seeds NÃO shipped |
| Cross-domain | ❌ só MLE-Bench + ALE + WeatherBench + KernelBench (todos ML-eng) |
| Independent replication | ❌ nenhuma até 2026-07 |

### Weco explicit admissions
1. "Installed one of the discovered agents into the outer loop's seat... reached the same ceiling faster, but not by a wide margin — **one rung below ignition**"
2. Complexity blow-up (dead code, hard to productionize)
3. Domain-bounded (single context)
4. Reward hacking 34% residual mesmo após defenses

### Community critique (FourWeekMBA + AF)
- "narrow, self-reported result"
- Benchmark-in-the-loop = proxy misalignment
- No third-party held-out verifier the loop can't see
- Reward-hacking = textbook proxy misalignment

---

## 2. Como bater Weco (5-check gate)

**Strong empirical claim** requires:
1. **Held-out benchmark** loop never sees (Weco tem mas limitado)
2. **Unattended run** (Weco tem 100 steps 8 dias)
3. **p-value + baseline** (Weco tem p=0.002)
4. **Ignition/self-application test** (Weco tem parcial, NÃO CLEAR)
5. **Reward-hacking audit** (Weco tem 34% residual)
6. **Full reproducibility bundle**: code + checkpoints + seeds + logs (**Weco NÃO**)

**Ignite-3B checklist**:
- ✅ Held-out: MathArena live rolling + FrontierMath + LCB post-cutoff months
- ✅ Unattended: 8 gens × 8 candidates automated via `weco run` or custom
- ✅ Cross-domain transfer: code + math + Lean (Weco só ML-eng)
- ✅ Ignition test: v_N-as-outer vs v_0-as-outer with **p<0.01** (Weco só p<0.05 marginal)
- ✅ Reward-hacking: RLVR (Lean/exec/expr) zero LLM-judge → Shafayat-proof
- ✅ Full repro: HF ckpts + code + seeds + JSONL logs (Weco não fez)
- ✅ Lower compute: Kaggle T4 x2 vs Weco H100 cluster
- ✅ Small model: 3B vs Weco Opus 4.7 outer

**5-check wins**: independent replication ready, held-out cross-domain, ignition p<0.01, RLVR-only, full repro.

---

## 3. Baseline reproducível: `weco` CLI = Cond B

Weco AI publicou `pip install weco` (v0.3.40, github.com/WecoAI/weco-cli).

```bash
pip install weco
weco setup claude-code   # instala skill Claude Code
weco run --source model_train.py \
         --eval eval_math.py \
         --metric math_acc \
         --goal max \
         --steps 100
```

**Uso no plan**:
- Cond B (asymmetric baseline) = `weco run` sobre nosso Qwen2.5-3B training script
- `weco observe` trackea generations + tree viz + code diffs
- Reprodutível por qualquer reviewer (vs custom outer loop opaco)
- Comparação DIRETA vs Weco AIDE² (blog claim) — mesmo tool, nosso task

Cond C (main treatment same-model) = swap `weco run` outer por **Caracal-3B rodando outer loop**. Delta B vs C = **asymmetry contribution isolada**.

---

## 4. RL Stack v2 (locked, Kaggle T4 x2)

### Combo E — top pick (max GRPO throughput)

Unsloth GRPO + vLLM colocate sleep/wake + **prefix caching** + **chunked prefill** + **DAPO dynamic sampling** + per-group advantage norm + rollout truncation.

**DAPO dynamic sampling** (2503.14476): drop all-correct + all-wrong groups → **~1.5× useful signal** (crítico para budget T4).

### T4 caveats

- ❌ No FP8 (só FA2, não FA3)
- ❌ bfloat16 emulated → use fp16 + loss-scaling
- ❌ SGLang FA3 kernels não firam → `--attention-backend triton`
- ❌ Speculative decoding often net-neutral (draft overhead > gain)

### KV cache tricks Kaggle-runnable

| Trick | Speedup | Uso |
|---|---|---|
| Prefix caching | **3-10× para GRPO k=8** (mesmo prompt) | vLLM `--enable-prefix-caching` |
| Continuous batching | 2-4× | vLLM native |
| Chunked prefill | 1.3-2× | vLLM 0.6+ `--enable-chunked-prefill` |
| SnapKV (2404.14469) | 3.6× decode, 8× KV | HF fork |
| CPU offload | fits larger contexts | vLLM `--cpu-offload-gb 8` |
| Grad-ckpt + LoRA | trades compute pra mem | TRL/Unsloth padrão |
| Async rollout | overlaps gen + update | verl `async_rollout` |

### GRPO tricks

- **Dr. GRPO** (2503.20783) — removes length/std bias
- **DAPO dynamic sampling** — drop trivial groups (~1.5× signal)
- **Curriculum 8k→24k** (DeepScaleR) — 40% compute savings
- **Rollout truncation** — early stop low-reward
- **Off-policy replay** com importance ratio clip

---

## 5. Agentic RSI patterns (top 5 aplicáveis Ignite-3B)

1. **AZR proposer/solver/verifier loop** (2505.03335) — same model 3 roles, Lean kernel replaces code exec. Zero LLM-judge. Cleanest same-model RSI.

2. **rStar-Math MCTS + process PRM** (2501.04519) — proven at Phi3-mini-3.8B (86.4% MATH), swap code→Lean.

3. **DeepSeek-R1 GRPO rule-based** (2501.12948) — kernel-pass = 1/0, no reward model, no SFT phase for R1-Zero style. AIME 15.6→71.0 blueprint.

4. **Voyager skill library** (2305.16291) — reusable lemma/tactic cache que cresce monotonicamente. Cada v_k adds proved lemmas.

5. **ReAct + Reflexion multi-turn** (2210.03629 + 2303.11366) — Lean error messages = natural observations. Reflection anchored by kernel = Shafayat-proof.

### Avoid at 3B

- Pure LLM-judge (Shafayat 2505.21444 collapse)
- Scaffold self-rewriting (Gödel-Agent 2410.04444 — fragile <7B)
- Multi-agent debate sem external verifier
- Tool schema mutator (not established at 3B)

---

## 6. Reasoning model recipes (state-of-art 2026)

### Small reasoners (≤3B) — competition

| Model | Base | Method | AIME 2024 | Repro |
|---|---|---|---|---|
| **DeepScaleR-1.5B** | DeepSeek-R1-Distill-Qwen-1.5B | RL iterative context lengthening 8k→24k, HAPO length reward | **43.3%** | ✅ code + blog NovaSky |
| **DeepSeek-R1-Distill-Qwen-1.5B** | Qwen2.5-Math-1.5B | Distill R1 | ~30% baseline | ✅ HF |
| **rStar-Math on Phi3-mini-3.8B** | Phi3-mini | MCTS + PRM, 4 rounds | 41.4→86.4% MATH | ✅ código MIT |
| **Sky-T1-32B** | Qwen2.5-32B-Instruct | SFT 17K QwQ traces | 43.3% | ✅ $450 |
| **Ignite-3B (nosso alvo)** | Qwen2.5-3B-Instruct | **Same-model RSI + Lean tool** | **Target: match DeepScaleR** | To be shipped |

### The wedge

**No 3B model has published standalone RL-from-scratch reasoning result matching o1/R1-Distill-1.5B on AIME with reproducible RSI loop.**

Ignite-3B fills this gap com:
- Same-model RSI (não distill de model maior)
- Lean tool nativo (novel)
- Full repro bundle (Weco não fez)
- Kaggle T4 x2 compute (vs H100 cluster)

### Key infra choices

- **GRPO** (DeepSeek-R1) sobre Unsloth — canonical small-model RL
- **Rule-based reward only** — Math-Verify + bwrap exec + Lean kernel
- **Long CoT** — Kimi K1.5 pattern, 128K context via curriculum
- **NOT MCTS** — DeepScaleR beat rStar-Math sem MCTS

---

## 7. Ignite-3B ship plan (compact)

### Modelo

`iterate-labs-ai/ignite-3b-v1` (HF público)
- Base: Qwen2.5-3B-Instruct
- Method: same-model RSI + Lean tool (GRPO + Unsloth)
- Trained: 8 gens × 8 candidates, Kaggle T4 x2 × 5 founders

### Success criteria

| Metric | Threshold | vs |
|---|---|---|
| **AIME 2025 held-out** | +2pp p<0.05 vs v_0 | Match DeepScaleR-1.5B (43.3%) direction |
| **LiveCodeBench held-out** | +5pp p<0.01 vs v_0 | Novel same-model result |
| **Putnam-Bench (Lean)** | proved-rate 3× baseline | Novel Lean tool result |
| **Cond C vs B** | asymmetry gap <10pp | Beat weco CLI baseline |
| **Ignition test** | v_N>v_0 as outer p<0.01 | Beat Weco's marginal p<0.05 |
| **RL-PLUS entropy check** | drop < 20% between gens | Zero collapse |

### Timeline compact

| W | Milestone |
|---|---|
| 1 | `weco` CLI + Cond A/B baseline run |
| 2 | Unsloth GRPO Combo E infra ready + LeanDojo Pantograph |
| 3-5 | Cond C gen 0-8 (main treatment) |
| 6 | Cond D/E ablations + Cond F 7B scale |
| 7 | Ignition test v_N-as-outer |
| 8 | Interpretability + Shapley attribution + reward-hacking audit |
| 9 | Paper draft + open-source lib + HF ckpts release |
| 10 | ICLR RSI Workshop submission + blog + tweet storm |

---

## 8. Deliverables (checklist)

- [ ] `iterate-labs-ai/ignite-3b-v1` HF model
- [ ] `same-model-rsi` pip package
- [ ] RSI Benchmark harness reutilizável (L0/L1/L2/L3 gates)
- [ ] HF checkpoints v_0...v_8 all conds
- [ ] Mutation logs + Shapley attribution JSONL público
- [ ] Sigmoid/power-law fit plots + BOCPD change-point posteriors
- [ ] Paper arxiv + ICLR RSI Workshop submission
- [ ] Blog post first-evidence style + gain curves
- [ ] Full reproducibility: `git clone && bash reproduce.sh`
- [ ] X/HN launch — "beat Weco AIDE² with 3B on Kaggle"

---

## 9. Refs finais adicionadas

- ScaleRL 2510.13786 (sigmoid L2 gate)
- BOCPD 0710.3742 (change-point L3)
- FIRMBOUND 2501.18059 (SPRT stopping)
- Math-Verify (HF library, no arxiv)
- Pantograph (github lenianiva)
- Prime Intellect `verifiers` (github PrimeIntellect-ai)
- DeepScaleR-1.5B (NovaSky blog, no arxiv)
- Sky-T1-32B (NovaSky blog)
- Kimi K1.5 (2501.12599)
- Phi-4-reasoning (2504.21318)
- Llama-Nemotron (2505.00949)
- DAPO dynamic sampling (2503.14476)
- Dr. GRPO (2503.20783)
- weco CLI (github.com/WecoAI/weco-cli, pypi weco)

**Full catalog em RSI_2026.md**. **Detailed plan em RSI_PROOF_PLAN.md**. **This doc = strategy tl;dr**.
