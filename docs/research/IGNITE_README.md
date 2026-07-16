# Ignite-3B - Same-Model Recursive Self-Improvement

Beats Weco AIDE² first-evidence claim with a 3B specialist model doing full
recursive self-improvement using itself as both inner (task solver) and outer
(scaffold mutator).

## Quick start (founders)

```bash
git clone -b s07-hybrid-agentic https://github.com/iterate-labs-ai/caracal-1
cd caracal-1
bash reproduce.sh   # end-to-end, ~120h Kaggle T4 x2
```

Or import Kaggle notebook by session (relay 5 founders × 12h):

| Session | Notebook | Purpose | Owner |
|---|---|---|---|
| S01 | `notebooks/kaggle/ignite/S01_baseline.ipynb` | Cond A eval v_0 + weco Cond B kick | Vitor |
| S02 | `S02_data_and_tools.ipynb` | Build 6 JSONLs + smoke tools | Kevin |
| S03 | `S03_gen0_2.ipynb` | Cond C outer loop gen 0-2 | Arthur |
| S04 | `S04_gen3_5.ipynb` | Cond C resume gen 3-5 | Vitor |
| S05 | `S05_gen6_8.ipynb` | Cond C resume gen 6-8, push v_1 final | Alexandre |
| S07 | `S07_ablation_D_E.ipynb` | LLM-judge ablation + frozen-inner | Pedro |
| S08 | `S08_cond_F_7b.ipynb` | Qwen2.5-7B scale ablation | Kevin |
| S09 | `S09_ignition_test.ipynb` | L2 gate: v_N as outer vs v_0 as outer | Arthur |
| S10 | `S10_shapley_writeup.ipynb` | Shapley attribution + audit + plots | Pedro |

## Architecture

```
Ignite-3B (Qwen2.5-3B-Instruct base)
  |
  Inner loop: solve task (math / code / Lean proof)
    | reward = RLVR (Math-Verify | bwrap exec | Lean kernel)
    | Shafayat-proof (no LLM-judge)
  |
  Outer loop: v_k proposes N=8 mutations to (prompt, scaffold, curriculum, LoRA cfg)
    | train 8 LoRA candidates via Unsloth GRPO + vLLM colocate
    | select top-2 by dev, verify on val (contamination gate)
    | retain if delta > 1pp AND entropy_ok AND passk_ok
  |
  Weight recursion: v_{k+1} = merge_lora(v_k, best_adapter)
```

**Novel**: v_k is BOTH inner and outer. Prior work (AIDE², DGM, HGM, SICA) uses
larger LLM (Opus/Gemini) as outer proposer + smaller LM as inner target.

## 5-condition experimental design

| Cond | Inner | Outer | Reward | Purpose |
|---|---|---|---|---|
| **A** | Qwen2.5-3B | (none) | n/a | Static baseline |
| **B** | Qwen2.5-3B | `weco run` (Opus 4.7-class) | RLVR | Asymmetric ceiling |
| **C** | **Qwen2.5-3B** | **Qwen2.5-3B (self)** | **RLVR** | **Main same-model RSI** |
| **D** | Qwen2.5-3B | Qwen2.5-3B (self) | LLM-judge | Shafayat collapse ablation |
| **E** | Qwen2.5-3B frozen | Qwen2.5-3B trainable | RLVR | Outer-only recursion |
| **F** | Qwen2.5-7B | Qwen2.5-7B (self) | RLVR | Scale ablation |

## Task suite (locked)

| Bench | Domain | RLVR verifier | Role |
|---|---|---|---|
| OMNI-MATH (2410.07985) | Math | Math-Verify | Main train + dev + val split |
| AIME 2024→2025 | Math | integer exact | Temporal split held-out |
| MathArena live | Math | Math-Verify | L3 gate zero-contam rolling |
| LiveCodeBench monthly (2403.07974) | Code | bwrap exec | L1 gate + L3 monthly buckets |
| BigCodeBench-Hard (2406.15877) | Code | bwrap exec | L2 asymptotic distinct dist |
| PutnamBench (2407.11214) | Math+Lean | Lean 4 kernel | Dual: eval + RL reward |

## Statistical gates

| Gate | Method | Threshold | File |
|---|---|---|---|
| **L1** | Held-out reward Δ bootstrap CI | Δ v_8 vs v_0 > 2pp p<0.05 | `stats.mcnemar_test` + `bootstrap_ci` |
| **L2** | Sigmoid fit R_∞ CI (ScaleRL 2510.13786) | Δ R_∞ 95% CI > 0 | `stats.sigmoid_fit` |
| **L3** | Bayesian change-point on cum_gain (0710.3742) | max posterior > 0.9 | `stats.bocpd` |
| **Stop** | Wald SPRT FIRMBOUND (2501.18059) | accept H0/H1 or continue | `stats.sprt_firmbound` |
| **Noise** | Variance-components ANOVA over 3 seeds | σ_seed / σ_within decomp | `stats.variance_components` |
| **Attribution** | Cumulative val_r credit per mutation | top-10 Shapley | `stats.shapley_attribution` |

## Success criteria (pre-registered)

- AIME 2025 held-out Δ v_8 vs v_0 ≥ **+2pp p<0.05**
- LiveCodeBench held-out Δ v_8 vs v_0 ≥ **+5pp p<0.01**
- Putnam-Bench (Lean) proved-rate ≥ **3× v_0 baseline**
- Cond C vs B asymmetry gap ≤ **10pp**
- Ignition test v_N > v_0 as outer ≥ **p<0.01**
- RL-PLUS entropy check drop **< 20%** between gens

**5-check gate to beat Weco**: (a) held-out cross-domain (b) unattended run
(c) p<0.01 (d) ignition test (e) reward-hacking audit (f) full repro bundle.

## Refs

- Weco AIDE² blog: https://weco.ai/blog/first-evidence-of-recursive-self-improvement
- Weco 4-level framework: https://weco.ai/blog/4-levels-of-recursive-self-improvement
- STOP (2310.02304), Gödel Agent (2410.04444), DGM (2505.22954), HGM (2510.21614)
- AZR (2505.03335), rStar-Math (2501.04519), SEAL (2506.10943)
- Shafayat 2505.21444 canonical collapse paper
- Zenil 2601.05280 theoretical grounding
- ScaleRL 2510.13786, FIRMBOUND 2501.18059
- Full catalog: `docs/research/RSI_2026.md`
- Detailed plan: `docs/research/RSI_PROOF_PLAN.md`
- Strategy tl;dr: `docs/research/RSI_STRATEGY.md`
