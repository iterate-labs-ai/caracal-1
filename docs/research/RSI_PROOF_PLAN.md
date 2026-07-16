# Research Plan: Same-Model Recursive Self-Improvement in a 3B Cybersec Specialist

## Working title
"Verifiable-Reward Recursive Self-Improvement in a Specialist Small Model:
Empirical Evidence at Weco Level 1 with a Path to Level 2"

## Motivation

Weco.ai definiu 4-level framework de RSI (L0-L3). Papers publicados atingem no
máximo L1 candidate (DGM, HGM, SICA, AIDE²). Nenhum sistema published passa
L2 (Weco recusou certificar próprio AIDE²).

**Todos os sistemas L1-candidate hoje usam LLM externo grande no outer loop**:

| Sistema | Outer (improver) | Inner (improved) |
|---|---|---|
| AIDE² | Claude Opus 4.7 | Gemini 3 Flash |
| DGM | Claude 3.5 API | Coding agents (API) |
| HGM | GPT-5-mini | Coding agents |
| SICA | Claude 3.5 API | Own code (API) |

**Model asymmetry**: outer é *sempre* maior/mais capaz que inner. Isso é **bootstrap**,
não recursão verdadeira. Um sistema realmente recursivo teria o MESMO modelo
fazendo ambos os papeis.

Gap concreto: **zero papers testaram same-model RSI num specialist pequeno com verifiable reward** (RLVR).

## Research questions

- **RQ1** (viability): Can a single 3B specialist model achieve **L1 net-positive RSI**
  on a cybersec task, using only itself as the outer-loop improver?
- **RQ2** (asymmetry): Does same-model recursion **match or approach** the asymmetric
  baseline (larger LLM as outer improver)?
- **RQ3** (asymptotic): Is the improvement **asymptotic** (v_N > v_0 forever) or
  **sample-efficient only** (fast start, plateaus)? This is the L2 gate.
- **RQ4** (stability): Does RLVR (verifiable reward) **prevent mode collapse**
  and reward hacking in same-model recursion? Weco AIDE² needed 3-layer defense.
- **RQ5** (interpretability): Can we characterize *which operators* the model
  discovers, addressing Weco AIDE²'s "very difficult to understand" limitation?

## Central hypothesis

> **H1**: A specialist small model (Caracal-3B) with hard-verifiable rewards
> (CWE match, CTF flag) can drive its own scaffold+prompt+data optimization over
> N ≥ 5 generations, producing gain-per-generation curves that fit a
> non-degenerate power law `gain = a·gen^(-b)` with `b < 0.5` (evidence of
> sustained learning, not immediate plateau).

Testable falsification: if `b ≥ 0.9` (essentially plateau after gen 1) or the
95% CI on `b` includes 1.0, we reject H1.

## Study design

### Conditions

| Cond | Inner | Outer | RLVR | Purpose |
|---|---|---|---|---|
| **A** | Caracal-3B | (none) | n/a | Static baseline |
| **B** | Caracal-3B | Claude Opus 4.8 API | Yes | Asymmetric bootstrap ceiling (AIDE² style) |
| **C** | Caracal-3B | Caracal-3B (self) | Yes | **Main treatment: same-model RSI** |
| **D** | Caracal-3B | Caracal-3B (self) | No (LLM-judge) | Ablation: RLVR necessity |
| **E** | Caracal-3B (frozen) | Caracal-3B (self, trainable) | Yes | Ablation: outer-only recursion |

### Task suite

**Primary**: CTI-Bench RCM (CVE description → CWE-NNN, arxiv 2406.07599).
- Objective: RLVR reward (exact CWE match = 1.0, ancestor = 0.5, sibling = 0.3, wrong = 0)
- Held-out: 2025-2026 CVEs (train on 2024 subset, eval on 2025 subset - **temporal split**)

**Secondary (generalization)**:
- CyberMetric-500 (unseen MCQ)
- Cybench-lite static CTF subset (unseen category shift)
- LiveCodeBench monthly (**non-cyber generalization**, arxiv 2403.07974)

### Recursion protocol (Cond C)

At generation `k`:
1. **Inference phase**: Caracal-3B(k) solves training CVEs, collect trajectories
2. **Introspection phase**: Same Caracal-3B(k) proposes N=8 candidate mutations to:
   - System prompt / instruction template
   - Chain-of-thought scaffold
   - Data augmentation strategy
   - LoRA rank / lr schedule (bounded)
3. **Evaluation phase**: Train N=8 candidate v(k+1) LoRAs (LoRA-mix cheap), eval each on held-in dev split with RLVR reward
4. **Selection phase**: Top-2 by dev reward, verified on unseen validation slice (contamination gate)
5. **Retention**: v(k+1) = argmax(unseen validation reward). If < v(k) + ε (ε=0.5pp), retain v(k).
6. Log everything: mutation type, dev delta, val delta, wall-clock, tokens.

**Total generations**: N=8 (Weco AIDE² used 7). Anchor v(0) = Caracal s05 checkpoint.

### Recursion protocol (Cond B - asymmetric baseline)

Same as C but step 2 uses Claude Opus 4.8 API (not Caracal-3B). Compares
same-model vs bigger-model outer.

## Metrics

### Primary

- **Held-out reward at generation N**: R_val(v_N) - R_val(v_0), bootstrap 95% CI
- **Gain-per-generation curve**: fit `gain(k) = a·k^(-b)`, report (a, b, R²)
- **v_N > v_0 as improver test (L2 gate)**: run v_N as outer on fresh v_0 candidate, does it produce better v_1 than v_0 does? McNemar paired test on held-out sample.

### Secondary

- **Generalization ratio**: gain on held-out non-cyber / gain on held-in cyber (should be > 0.5 for genuine recursion, not overfit)
- **Wall-clock $ per generation**: canonical Weco budget (compute + API cost)
- **Reward-hacking rate**: fraction of candidates that gamed dev reward but failed val
- **Mode collapse detector**: entropy of generated trajectories per generation
- **Operator attribution**: which mutation types contribute most gain?

### Statistical rigor

- Bootstrap CI (n=10K resamples)
- McNemar paired test (per sample, matched design)
- Bonferroni correction across conditions (α = 0.05 / 5 = 0.01)
- Power analysis: N=1000 samples yields 80% power to detect δ=2pp @ α=0.01

## Compute budget

**Kaggle T4 x2 relay** (5 founders × 12h × 3-4 sessões = 180-240h total)

| Session | Owner | Task | Hours |
|---|---|---|---|
| S01 | Pedro | Cond A + B baseline | 12h |
| S02 | Kevin | Cond C gen 0-2 | 12h |
| S03 | Arthur | Cond C gen 3-5 | 12h |
| S04 | Vitor | Cond C gen 6-8 | 12h |
| S05 | Alexandre | Cond D ablation | 12h |
| S06 | Pedro | Cond E ablation | 12h |
| S07 | Kevin | Cond C L2 test (v_N as outer) | 12h |
| S08 | Arthur | Held-out eval all conds | 12h |
| S09 | Vitor | Ablations + interpretability | 12h |
| S10 | Alexandre | Final writeup + figures | 12h |

Total: **120h Kaggle** (dentro budget 5 founders × 30h/sem = 150h/sem).
Anthropic API budget (Cond B only): **$300 estimado** (Opus 4.8 outer, 8 gens × 8 candidates × 5K tokens each).

## Novel contributions (sharpened after gap analysis)

1. **First same-3B-weights RSI** as both cyber-solver AND scaffold-mutator. STOP/DGM/DARWIN
   use frontier models as mutators; AZR/R-Zero share weights but only at task-policy
   level, not scaffold-code level. No prior work at 3B in cyber domain.

2. **Sigmoid-fit L2 gate** (ScaleRL 2510.13786 methodology) - separate *asymptotic reward*
   from *sample efficiency parameter* per outer iteration. No RSI paper currently reports
   this decomposition. Cleaner than Weco's ad-hoc "asymptotic" claim.

3. **Nested-loop noise decomposition** - variance-components ANOVA over replicated ignition
   runs: `var(inner-gain) = σ²_seed + σ²_data + σ²_mutation + σ²_outer_eval`. First formal
   treatment. Combined with FIRMBOUND SPRT (2501.18059) outer-loop stopping rule and
   Bregman-conforming metric (Heskes 2501.18581) to guarantee decomposition holds.

4. **Shape-agnostic L3 estimator** - Bayesian model selection between `gain = a·t^(-b)`
   (power-law diminishing) and `gain = c·exp(kt)` (exponential accelerating) + online
   Bayesian change-point (Adams-MacKay 0710.3742) on `d²(cum_gain)/dt²`. Preregister
   b<0 / k>0 threshold *conditional on measured noise floor*.

5. **Direct empirical test of asymmetry hypothesis**: does inner==outer capability
   actually prevent lift (as Shafayat 2505.21444 predicts), or does verifiable cyber
   reward break collapse (as AZR shows for code)?

6. **Contraction-rate measurement** on outer-loop operator (Zenil 2601.05280 theory,
   nobody measured on real system yet).

7. **Mutation-attribution tree**: Shapley-value credit per operator on final gain +
   semantic-diff clustering across generations. Addresses AIDE²'s stated open problem
   "very difficult to understand how system works." No published equivalent.

8. **Failure catalog at 3B**: which outer edits collapse the model (RL-PLUS-style),
   which extrapolate (Lee/Papailiopoulos-style). Fills empirical gap under Zenil theory.

9. **Cybersec first**: temporal CVE split (2024 train → 2025 held-out) - cleaner than
   random split, no prior RSI in security domain.

## Statistical machinery (locked in)

| Component | Method | Reference |
|---|---|---|
| L2 asymptotic vs sample-efficient | Sigmoid fit per outer iter, Δasymptote CI test | ScaleRL 2510.13786 |
| L3 detection | Bayesian model selection power-law vs exponential | 2509.09677 + 0710.3742 |
| Outer-loop stopping | SPRT with overshoot correction | FIRMBOUND 2501.18059 + 2410.16076 |
| Mutation selection | UCB-style regret bounds | TextBO 2511.12063 + Evo-MAB 2205.10113 |
| Nested-loop noise | Variance-components ANOVA (seed, data, mutation, eval) | Signal&Noise 2508.13144 |
| Bootstrap CI | Cheap subsampling | 2501.10289 |
| Metric choice | Bregman divergence required for bias-var split | Heskes 2501.18581 |
| Hidden-consistent eval | AIRA_2 protocol to fight overfitting-to-val | 2603.26499 |
| Per-operator attribution | Gradient fingerprints + Shapley values | 2604.16242 |
| Contamination guard | Threshold 1.2-5% based on reward-hacking onset | 2505.21444 |

## Related work positioning (refined)

| Paper | Contribution | Vs Ours |
|---|---|---|
| **STOP** (2310.02304, COLM'24) | Single GPT-4 improves scaffold. Authors admit "not full RSI - LM itself not altered." | Ours: SEAL-style weight edits + scaffold mutations combined. |
| **Gödel Agent** (2410.04444, ACL'25) | Single LLM rewrites own scaffold at runtime. Game-of-24 4→78%. | Closest architecture. Ours: weight recursion, cybersec RLVR. |
| **AIDE²** (2502.13138) | Asymmetric outer (Opus 4.7 → Gemini-Flash inner). 7 versions, +0.053 MLE-Lite p=0.0024. | Ours: same-model, cybersec, RLVR, temporal split. |
| **DGM** (2505.22954) | Archive-based, API models. SWE 20→50. | Ours: single trajectory, no external LLM. |
| **HGM** (2510.21614, ICLR'26 oral) | CMP metric, GPT-5-mini. | Ours: power-law fit + explicit asymptotic test. |
| **AZR** (2505.03335) | Same-model RLVR, general reasoning, zero data. | Ours: specialist domain + operator attribution. |
| **R-Zero** (2508.05004, ICLR'26) | Challenger/Solver same base but *separate weights*. | Ours: single weight trajectory. |
| **Self-Improving Transformers** (2502.01612) | OOD arithmetic, tiny model. Linearly-growing OOD generalization. | Ours: production 3B specialist, cyber. |
| **rStar-Math** (2501.04519) | Phi3-mini 3.8B + Qwen2.5-Math-7B, 4 rounds MCTS+PRM. | Ours: no MCTS, direct scaffold+weight edits. |
| **SEAL** (2506.10943, MIT) | Self-edits: LM emits SFT data + hparams. Weight-level. | Closest weight-level precedent. Ours: adds scaffold+prompt layer. |

## Failure-mode papers to cite AND beat

| Paper | Failure documented |
|---|---|
| **Can LRMs Self-Train?** (2505.21444, Shafayat) | Majority-vote self-reward → sudden complete collapse via reward hacking. Canonical failure paper. |
| **RL-PLUS: Capability Boundary Collapse** (2508.00222) | Entropy collapse + support shrinkage under RLVR. |
| **Beyond Pass@1** (2508.14029) | pass@1↑ / pass@k↓ tradeoff under self-play. |
| **Zenil: Limits of Self-Improving** (2601.05280) | Formalizes RSI as dynamical system, proves 2 failure modes (entropy decay, variance amplification) when exogenous signal vanishes. **Theoretical grounding**. |
| **Task-Centric Theory** (2602.10014) | Formal conditions under which iterative self-improvement is sustained. Closest to L3 predictor. |

**Key rebuttal**: our RLVR reward (CWE match / CTF flag) is exogenous verifiable — Zenil's collapse conditions don't apply. Test empirically.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| 3B model too weak pra propor mutations úteis | Cond B (asymmetric) mostra ceiling; se gap grande, honest failure result |
| Mode collapse / reward hacking | Cond D ablation isola RLVR contribution |
| Contamination train/eval | Temporal split (CVE year), held-out non-cyber (LiveCodeBench monthly) |
| Kaggle 12h kernel limit | Checkpoint per generation, 1 gen per session |
| Statistical power insuficiente | Power analysis pre-registered, N=1000 held-out samples |
| Overfitting a CTI-Bench | 4 held-out benches (CyberMetric, Cybench-lite, LiveCodeBench, temporal split) |

## Deliverables

1. **Paper draft** (ICLR/NeurIPS 2027 submission target): 9 pages + appendix
2. **Reproducible code** (branch `s07-hybrid-agentic` extended)
3. **HF checkpoints** for v_0...v_N em cada condition
4. **Data release**: mutation logs + attribution analysis (public JSONL)
5. **Ablation tables + power-law fit plots** (matplotlib, seaborn)

## Timeline

| Semana | Milestone |
|---|---|
| W1 | Cond A + B baseline done, power analysis validated |
| W2-W4 | Cond C gen 0-8, log everything |
| W5 | Cond D + E ablations |
| W6 | Held-out eval all conditions |
| W7 | L2 test (v_N as outer) |
| W8 | Interpretability + writeup |
| W9 | Reviewer pass (external), submit |

**Total: 9 semanas** (~2.5 meses) do Sprint 1 start ao submission.

## Success criteria (pre-registered)

- **Primary**: Cond C v_8 held-out RCM >= Cond A + 3pp with p < 0.01 → **L1 confirmed**
- **Secondary**: Cond C generalization ratio > 0.5 → genuine recursion
- **Aspirational**: v_N > v_0 as outer improver with p < 0.05 → **L2 candidate**
- **Failure mode acceptable**: null result on L1 is publishable (asymmetry required)

## Open technical questions (research agenda)

1. What's the minimal specialist model capacity for same-model RSI?
2. How does verifier hardness (RLVR vs LLM-judge) affect stability?
3. Do mutation operators discovered by 3B generalize to 8B / 70B?
4. Can we detect L3 signal at Kaggle scale? (probably no - budget too small,
   but publish the negative + noise bound)
5. What's the phase transition (if any) between L0 and L1 for specialist models?

---

## References base

Todas em `docs/research/RSI_2026.md` (papers catalog + open-source + eval metrics).

Cite ordering priority (updated):
1. Weco 4-levels + first-evidence blog posts
2. STOP 2310.02304 (gap statement: "not full RSI")
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

**UNVERIFIED (checar antes de citar)**: CyberEvolver 2605.26195 (cyber neighbor claim),
Self-Reference Introspection 2607.04277, Self-Play Only Evolves 2603.02218.
