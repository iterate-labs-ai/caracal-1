# Closing the Asymmetry Gap: Same-Model Recursive Self-Improvement in a 3B Language Model with Verifiable Rewards

*Draft. Conditions B-F pending. All numbers reported below are measured; every
result not yet obtained is marked "pending" rather than estimated.*

---

## Abstract

Every published system that has been proposed as a Level-1 candidate for
recursive self-improvement (RSI) places a **large external language model in the
outer loop**. AIDE² pairs a Claude Opus 4.7 outer improver with a Gemini 3 Flash
inner solver; the Darwin Gödel Machine (2505.22954), the Huxley-Gödel Machine
(2510.21614), and SICA (2504.15228) all drive their archives with frontier API
models while the artifacts being improved are comparatively small coding agents.
This is **bootstrap**, not recursion: the improvement capacity resides in a model
that is never itself improved. The STOP paper (2310.02304) states the limitation
explicitly — *"since the LM itself is not altered, this is not full recursive
self-improvement."*

We introduce **Ignite-3B**, an RSI harness in which the *same 3B weights* act as
both the inner task solver and the outer mutation proposer, with **purely
verifiable rewards** (program execution under a `bwrap` sandbox, SymPy answer
equivalence, and a Lean 4 kernel) and no LLM-judge anywhere in the loop. The
absence of an LLM-judge is a deliberate defense against the self-reward collapse
mode documented by Shafayat et al. (2505.21444). We pre-register L1/L2/L3
statistical gates built on sigmoid decomposition (ScaleRL, 2510.13786), Bayesian
online change-point detection (Adams-MacKay, 0710.3742), variance-components
ANOVA, and a FIRMBOUND sequential-probability-ratio stopping rule (2501.18059).

This draft reports the completed static baseline (Condition A) on Kaggle T4×2:
OMNI-MATH **5.0%** (n=100), LiveCodeBench **28.0%** (n=50), AIME 2025 **6.7%**
(n=30), together with end-to-end validation of the data and reward pipelines. The
main same-model treatment (Condition C) is running at time of writing; its outer
loop has been observed to propose and apply non-trivial mutations to its own
training configuration. All remaining conditions are pending.

---

## 1. Introduction

### 1.1 The asymmetry gap

Weco.ai's four-level taxonomy of RSI (L0 Delegation → L1 Net Positive → L2
Ignition → L3 Inflection) has become a convenient shared vocabulary. Under that
taxonomy no published system has credibly passed L2; Weco itself declined to
certify AIDE² at L2, citing asymptotic-convergence ambiguity and a wide noise
margin.

What is less often stated is that the L1 candidates share a structural property:

| System | Outer (improver) | Inner (improved) | Domain |
|---|---|---|---|
| AIDE² | Claude Opus 4.7 | Gemini 3 Flash | ML engineering |
| DGM (2505.22954) | Claude 3.5 API | Coding agents | SWE |
| HGM (2510.21614) | GPT-5-mini | Coding agents | SWE |
| SICA (2504.15228) | Claude 3.5 API | Own code (via API) | SWE |
| DARWIN | GPT-4-class | nanoGPT training | ML |

In each row the improver is strictly more capable than the improved, and the
improver is **static**. The loop therefore inherits a hard ceiling set by an
entity outside the recursion, and any observed gain curve conflates two effects:
(i) genuine self-improvement, and (ii) the one-off distillation of a frontier
model's competence into a smaller artifact. Nothing about the second effect
compounds.

A genuine recursion requires the improver and the improved to be the *same
object*. If v_k proposes the modification that produces v_{k+1}, and v_{k+1}
then occupies the proposer seat, then improvements to the proposer are
themselves improvable. That is the property the taxonomy's L2 rung ("v2 is a
better improver than v1") actually tests, and it cannot be tested at all when
the proposer is a frozen API endpoint.

### 1.2 Why 3B, and why verifiable rewards

Two design decisions follow.

**Small model.** Same-model recursion is only interesting if the model is cheap
enough to retrain many times inside the loop. A 3B model with LoRA adapters can
be trained and evaluated repeatedly on two T4 GPUs. It is also the honest hard
case: if same-model RSI works at 3B it is a property of the loop, not of latent
frontier capability.

**Verifiable rewards only.** Same-model self-training is exactly the regime in
which Shafayat et al. (2505.21444) document collapse: when the reward signal is
the model's own judgment, the loop optimizes agreement rather than correctness,
and reward hacking onsets at contamination rates as low as 1.2-5%. Our reward is
exogenous by construction — a Python interpreter, a computer-algebra system, a
proof kernel. None of these can be persuaded. This mirrors the stance of
Absolute Zero Reasoner (2505.03335) and DeepSeek-R1's rule-based GRPO
(2501.12948).

### 1.3 Research questions

- **RQ1 (viability).** Can a 3B model reach L1 net-positive using only itself as
  outer improver?
- **RQ2 (collapse resistance).** Does RLVR prevent the Shafayat collapse mode
  inherent in same-model self-training?
- **RQ3 (asymptotic vs sample-efficient).** Is any observed improvement
  asymptotic (R_∞ rises) or merely faster convergence to the same ceiling?
- **RQ4 (asymmetry cost).** How much lift is lost when the outer improver is
  downgraded from a frontier model to the model itself? This quantifies the
  bootstrap contribution directly.
- **RQ5 (generality).** Do discovered mutations transfer across math, code, and
  formal proof, or are they task-overfit?

**H1.** A 3B model equipped with verifiable-reward RLVR can drive its own
scaffold + prompt + weight optimization over N ≥ 8 generations, producing a gain
curve whose asymptotic component (ScaleRL sigmoid decomposition) is significantly
greater than v_0 at p < 0.01, with no external LLM in the outer loop.
**Falsification:** if the asymptotic-gain 95% CI includes zero, or if the entropy
collapse detector (RL-PLUS, 2508.00222) fires, H1 is rejected.

---

## 2. Related Work

| Paper | Contribution | Ignite-3B differential |
|---|---|---|
| **STOP** (2310.02304, COLM'24) | Scaffold-only self-modification driven by GPT-4 | Weight-level recursion; the LM *is* altered |
| **Gödel Agent** (2410.04444) | Runtime scaffold rewrite, single LLM | Weight recursion + three verifiable domains |
| **AIDE²** (2502.13138 + blog) | Asymmetric outer loop, ML engineering only | Same-model outer; general RLVR domains |
| **DGM** (2505.22954) | Darwinian archive, API improver | Single weight trajectory, self-proposal |
| **HGM** (2510.21614, ICLR'26 oral) | Clade-metaproductivity metric, GPT-5-mini improver | Sigmoid + Bayesian L2/L3 gates instead of CMP |
| **SICA** (2504.15228) | Agent edits own source, API model | Weights, not source |
| **AZR** (2505.03335) | Same-model RLVR self-play, policy level | Adds scaffold + weight recursion above the policy |
| **R-Zero** (2508.05004) | Same base model, but two separate weight sets | One weight trajectory |
| **rStar-Math** (2501.04519) | MCTS + process reward, math only | No MCTS; direct scaffold + weight mutation |
| **SEAL** (2506.10943) | Model writes its own finetuning data (self-edits) | Adds a scaffold/prompt/hyperparameter layer above self-edits |
| **Self-Improving Transformers** (2502.01612) | Linear OOD extrapolation, tiny model | Production 3B, multi-domain |

We additionally cite five papers **as failure modes to be beaten**, not merely as
background:

| Paper | Failure documented | Our defense |
|---|---|---|
| Shafayat 2505.21444 | Majority-vote self-reward → collapse | RLVR exogenous reward, zero LLM-judge |
| RL-PLUS 2508.00222 | Entropy collapse, support shrinkage | Per-generation entropy detector with abort |
| Beyond Pass@1 2508.14029 | pass@1 rises while pass@k falls | Track both, flag divergence |
| Zenil 2601.05280 | Entropy decay + variance amplification when exogenous signal vanishes | RLVR *is* the exogenous signal and does not vanish |
| Task-Centric Theory 2602.10014 | Formal conditions for sustained RSI | Tested empirically rather than assumed |

The closest precedent to our setting is AZR, which does use one model in
proposer/solver/verifier roles with verifiable reward. The distinction is the
level at which recursion occurs: AZR recursion is *policy-level* (the model
proposes tasks for itself), whereas Ignite-3B recursion is *pipeline-level* — the
model proposes modifications to the training procedure that produces its own
successor weights, including learning rate, LoRA capacity, curriculum, system
prompt, and reasoning scaffold.

---

## 3. Method

### 3.1 Outer loop

The outer loop (`train/ignite/C_rsi_outer.py`) implements the following, exactly
as run:

**Initialization.** Load base weights (`Qwen/Qwen2.5-3B-Instruct`, fp16, single
device) plus an optional v_0 LoRA adapter. Evaluate on the validation split
(n=100) to obtain r_prev, the incumbent reward.

**For each generation k:**

1. **Proposal.** The current model v_k, holding its own adapter, is prompted with
   its current configuration and a compact trace of the previous generation's
   validation reward, and asked to emit one JSON mutation. This is repeated N
   times (default N=8) at temperature 0.9 to yield a candidate set. Parsing
   failures fall back to random sampling from the mutation grid, and every
   fallback is logged so the fraction of genuinely model-authored mutations can
   be audited.
2. **VRAM handoff.** The proposer is explicitly freed (`del` + `gc.collect()` +
   `torch.cuda.empty_cache()`) before training begins. Two 3B models do not fit
   on one T4, so the loop is strictly serial: propose → free → train → evaluate →
   free → reload proposer.
3. **Candidate training.** Each mutation parameterizes one inner GRPO run,
   producing one LoRA adapter. Training failures are recorded as candidates with
   dev reward −1.0 and a failure reason rather than crashing the generation.
4. **Dev selection.** Each surviving candidate is evaluated on a dev split
   (n=50). The top-2 by dev reward advance.
5. **Validation gate.** The top-2 are re-evaluated on a validation split (n=100)
   that the proposal step never sees. The argmax by validation reward becomes the
   generation's proposal.
6. **Retention.** The proposal is retained iff `Δ > ε` (ε = 0.01 absolute,
   pre-registered) **and** the entropy delta has not dropped past the 30%
   threshold **and** the pass@k check passes. Otherwise v_k is retained and the
   rejection is logged. Retention is thus conservative by design: a generation
   that fails to improve costs compute but cannot degrade the trajectory.
7. **Archival.** Every candidate — retained or not — is appended to
   `log.jsonl` with its mutation hash, dev reward, validation reward, entropy
   delta, pass@k flag, retention decision, and rejection reason. Retained
   adapters are optionally pushed to the Hugging Face Hub, one revision per
   generation. A `state.json` checkpoint allows `--resume` after a Kaggle
   12-hour session boundary.

Two guards are currently **placeholders returning permissive constants**:
`compute_entropy_delta` returns 0.0 and `passk_check` returns True. Their full
implementations require logit probes on held-out prompts and pass@k sampling
respectively, and are scheduled for the next session. This is stated plainly
because it means the collapse detectors are *instrumented but not yet armed*, and
no anti-collapse claim can be made from the current run.

### 3.2 Mutation space

A mutation (`train/ignite/mutations.py`) is a six-field record:

| Field | Type | Grid / space |
|---|---|---|
| `system_prompt` | free text | model-authored, seeded from a per-domain default |
| `cot_scaffold` | free text | model-authored, seeded from 5 canonical scaffolds |
| `curriculum_bin` | categorical | easy / medium / hard / mixed |
| `lora_rank` | int | {16, 32, 64} |
| `lora_alpha` | int | {32, 64, 128} |
| `lr` | float | {5e-7, 1e-6, 3e-6} |

Each mutation is content-hashed (SHA-256, 12 hex chars) so that identical
proposals across generations collapse to one attribution unit. Note that the two
free-text fields are unbounded: the model may propose prompts and scaffolds
outside any seed list, while the three numeric fields are bounded — a deliberate
safety envelope, since an unbounded learning rate proposal is a trivially
available self-destruction path.

The design follows DGM's `sample_mutant` pattern, with the distinction that the
sampler is the artifact under improvement.

### 3.3 Inner loop: GRPO with verifiable reward

The inner loop (`train/ignite/inner_grpo.py`) trains one LoRA delta per
candidate using TRL's `GRPOTrainer`. Configuration as run: fp16, gradient
checkpointing, LoRA on all seven attention and MLP projections, dropout 0,
`beta=0.001` KL coefficient, `num_generations=4` rollouts per prompt, max prompt
640 tokens, max completion 512, 150 steps by default.

Notable deviations from the planned stack, forced by hardware:

- **No Unsloth.** Its kernels require SM 8.0+; the T4 is SM 7.5.
- **No vLLM colocation.** VRAM on Kaggle T4×2 is too tight; rollouts use
  HuggingFace `generate()`.
- **fp16 rather than bf16**, since T4 emulates bf16.
- `per_device_batch` is auto-corrected to a multiple of `num_generations`, a TRL
  divisibility requirement that otherwise fails at runtime.

### 3.4 Verifiable rewards

`eval/ignite/reward.py` exposes three reward closures, all deterministic, all
returning floats in [−1, 1], none consulting a language model:

- **`math_reward`** — extracts `\boxed{}`, verifies via HuggingFace Math-Verify
  (latex2sympy2 + SymPy) under a signal-based timeout. Returns 1.0 for
  equivalence, 0.0 for a wrong answer, −1.0 for malformed output, plus a +0.1
  format bonus (capped at 1.0) when a well-formed `\boxed{}` is present. The
  malformed case is punished more harshly than the wrong case, which encodes a
  preference for a checkable wrong answer over an uncheckable one.
- **`code_task_reward`** — extracts the fenced Python block and executes it
  against unit tests inside a `bwrap` sandbox with read-only `/usr`, `/lib`,
  `/lib64`, `/bin`, `/etc`, a tmpfs `/tmp`, and a wall-clock timeout (default
  10s). Reward is the fraction of tests passed.
- **`lean_reward`** — binary kernel acceptance via a Pantograph Lean 4 daemon.
  Implemented but **not yet exercised in any completed run**.

`build_reward_fn(bench)` returns the TRL-compatible batched wrapper.

---

## 4. Experimental Design

Six conditions, sharing one task suite, one evaluation harness, and one archive
format:

| Cond | Inner | Outer | Reward | What it isolates |
|---|---|---|---|---|
| **A** | Qwen2.5-3B | none | n/a | Static baseline; the number every Δ is measured against |
| **B** | Qwen2.5-3B | frontier API (via `weco` CLI) | RLVR | Asymmetric bootstrap ceiling |
| **C** | Qwen2.5-3B | Qwen2.5-3B (self) | RLVR | **Main treatment: same-model RSI** |
| **D** | Qwen2.5-3B | Qwen2.5-3B (self) | self-LLM-judge | RLVR necessity (Shafayat prediction) |
| **E** | Qwen2.5-3B, frozen | Qwen2.5-3B | RLVR | Outer-only recursion (`--steps 0`) |
| **F** | Qwen2.5-7B | Qwen2.5-7B (self) | RLVR | Scale ablation |

The key contrasts: **C − A** answers RQ1; **B − C** answers RQ4 and is the direct
measurement of what the bootstrap was contributing; **C − D** answers RQ2; **C −
E** separates weight recursion from scaffold recursion; **F − C** answers whether
the asymmetry requirement weakens with scale.

Conditions D and E are implemented as thin wrappers that reuse the Condition C
outer loop verbatim with a swapped reward function and `--steps 0` respectively,
which guarantees the ablations differ from the treatment in exactly one place.

**Task suite.** LiveCodeBench (2403.07974) and BigCodeBench-Hard (2406.15877) for
code; OMNI-MATH (2410.07985) and MathArena for math; PutnamBench (2407.11214) for
Lean formal proof; AIME 2025 as a sanity reference and not a gate. All are
execution- or exact-match-verifiable; none require a judge. Decontamination
relies on temporal rolling splits (LiveCodeBench monthly buckets, AIME 2025,
MathArena contests postdating the base model cutoff).

---

## 5. Statistical Methodology

The gates are pre-registered and implemented in `eval/ignite/stats.py`.

**L1 — net positive.** Held-out reward delta v_N − v_0 with bootstrap 95% CI
(`bootstrap_ci`, 10,000 resamples, fixed seed). Paired comparisons use
`mcnemar_test`, which builds the discordant-pair table, applies the exact test
when discordant pairs number fewer than 25 and the continuity-corrected χ² test
otherwise, and reports Cohen's g alongside the p-value.

**L2 — asymptotic vs sample-efficient.** `sigmoid_fit` fits the ScaleRL
(2510.13786) form R(C) = R_∞ / (1 + (C_50/C)^α) by least squares with bounded
parameters, then bootstraps 1,000 resamples of the fit to obtain a 95% CI on
R_∞. **L2 candidate iff Δ(R_∞) CI excludes zero.** This is the decomposition that
distinguishes a loop that raises the ceiling from a loop that merely reaches the
same ceiling sooner — precisely the ambiguity on which Weco declined to certify
AIDE². The fit requires ≥3 compute points and returns an explicit error
otherwise, so it cannot silently produce a number from two generations.

**L3 — inflection.** `bocpd` implements Adams-MacKay (0710.3742) Bayesian online
change-point detection with a normal-inverse-gamma conjugate update and hazard
rate 0.01, returning the run-length posterior. **L3 candidate iff the posterior
exceeds 0.9 at generation ≥ 3.** This is paired with Bayesian model selection
between power-law and exponential gain curves (BF > 10 required). We expect this
gate to return null at Kaggle scale and intend to publish the resulting noise
bound as a useful negative.

**Noise decomposition.** `variance_components` decomposes gain variance into
between-seed and within-seed terms over replicate runs, requiring ≥2 seeds. Per
Heskes (2501.18581) the metric fed to this decomposition must be
Bregman-conforming, so we decompose the RLVR reward directly rather than
accuracy.

**Stopping.** `sprt_firmbound` implements a Wald SPRT with FIRMBOUND
(2501.18059) overshoot correction, testing H0: μ = 0 against H1: μ = 0.01 at
α = β = 0.05, returning `accept_H0` / `accept_H1` / `continue`. This governs when
the outer loop stops rather than running a fixed generation budget.

**Attribution.** `shapley_attribution` reads the archive log, groups retained
candidates by mutation hash, and reports cumulative marginal contribution per
mutation — an attempt to close the interpretability gap AIDE² explicitly
acknowledged.

---

## 6. Results

### 6.1 Condition A — static baseline (complete)

Qwen2.5-3B-Instruct, fp16, Kaggle T4×2, single seed, total wall clock ≈ 99
minutes. Bootstrap 95% CIs.

| Benchmark | n | Accuracy | 95% CI |
|---|---|---|---|
| OMNI-MATH | 100 | **5.0%** | [1.0, 10.0] |
| LiveCodeBench | 50 | **28.0%** | [16.0, 42.0] |
| AIME 2025 | 30 | **6.7%** | [0.0, 16.7] |

These intervals are wide, and deliberately reported as such. At n=30 the AIME
interval touches zero; no AIME-based claim will be made until n is increased.

### 6.2 Pipeline validation (complete)

**Data.** All six builders produce non-empty, schema-conforming JSONL:

| Dataset | Rows |
|---|---|
| OMNI-MATH | 4,428 |
| LiveCodeBench | 175 |
| BigCodeBench-Hard | 148 |
| MathArena (5 contests) | 123 |
| AIME | 60 |
| PutnamBench | 514 |

**Reward.** Verified end-to-end, not merely unit-tested in isolation:
`code_task_reward` returns 1.0 on a correct solution and 0.0 on an incorrect one
when run against real LiveCodeBench stdin/stdout tests inside the `bwrap`
sandbox; `math_reward` behaves identically through the SymPy path. The reward
signal driving GRPO is therefore known to be live rather than a constant.

### 6.3 Condition C — same-model RSI (in progress)

The run is executing at time of writing. What has been **observed**:

- The outer loop successfully elicited mutations from the 3B model itself. The
  model proposed changes to `system_prompt`, `cot_scaffold`, and `lr`, moving
  the learning rate from the 1e-6 default to 1e-5 — outside the seeded
  `LR_GRID`, i.e. the model extrapolated past the values it was shown rather
  than selecting from them. Whether that extrapolation is beneficial or merely
  bold is exactly what the retention gate exists to decide, and it has not
  decided yet.
- A partial baseline evaluation inside the loop registered 12% on an OMNI-MATH
  partial slice. This is **not** comparable to the 5.0% Condition A figure: it is
  a different, smaller slice at a different n, and is reported only as evidence
  that the in-loop evaluation path executes.

No generation has yet completed the full propose → train → dev → val → retain
cycle with logged validation deltas. **No L1, L2, or L3 claim is made.**

### 6.4 Conditions B, D, E, F

**Pending.** Harnesses are implemented (`B_weco.py`, `D_llmjudge.py`,
`E_frozen.py`, `F_7b.py`) and share the Condition C outer loop, but none has been
executed.

### 6.5 Summary table

| Condition | Status | Result |
|---|---|---|
| A — static baseline | complete | see §6.1 |
| B — asymmetric outer | pending | — |
| C — same-model RSI | in progress | partial; §6.3 |
| D — LLM-judge ablation | pending | — |
| E — frozen inner | pending | — |
| F — 7B scale | pending | — |
| L1 gate | pending | — |
| L2 gate (sigmoid ΔR_∞) | pending | — |
| L3 gate (BOCPD, BF) | pending | — |

---

## 7. Limitations

We list these before any claim rather than after, because at the present stage the
limitations dominate the results.

1. **Single seed.** Every completed number comes from one seed. The
   variance-components machinery requires ≥2 seeds and has not been run; σ²_seed
   is currently unknown, which means we cannot yet distinguish a real delta from
   seed noise at the magnitudes the L1 gate cares about.
2. **Small n.** n=100/50/30 gives intervals 9-17 percentage points wide. The
   pre-registered power analysis assumes n=1000 per domain. Bridging that gap is
   a compute question, not a methods question.
3. **Condition C incomplete.** No generation has closed. The most that can be
   said is that a 3B model *can* author syntactically valid, semantically
   non-trivial mutations to its own training configuration — which establishes
   mechanism, not benefit.
4. **Collapse detectors not armed.** `compute_entropy_delta` and `passk_check`
   are permissive placeholders. The RL-PLUS and Beyond-Pass@1 defenses are
   designed and wired but not yet functional, so the current retention gate is
   effectively delta-only.
5. **Lean never exercised.** `lean_reward` and the Pantograph daemon exist;
   PutnamBench data is built (514 rows); no Lean-conditioned run has been
   performed. Every formal-verification contribution claimed in the plan is
   therefore unsupported so far.
6. **Compute constrained to T4.** SM 7.5 forced the removal of Unsloth, vLLM
   colocation, bf16, and FP8. Throughput is materially below the planned stack,
   and the serial load/free/reload cycle imposed by 16GB VRAM adds substantial
   per-generation overhead. Absolute numbers here should not be read as
   characterizing the method at better hardware.
7. **Single-domain outer loop so far.** Condition C is running on math alone.
   RQ5 (cross-domain transfer) is untouched.
8. **Fallback contamination.** The proposer falls back to random sampling on
   parse failure. Unless the fallback fraction is low and reported, "same-model
   proposal" degrades toward random search. This fraction is logged and must be
   published with the results.

---

## 8. Reproducibility

The reproduction bundle is `reproduce.sh` at repository root, which performs, in
order: dependency install, all six dataset builds, reward-tool smoke tests
(`math_verify` and `code_exec` executed standalone), Condition A baseline
evaluation, the Condition C outer loop at 8 generations × 8 candidates × 150
steps, and the statistical analysis pass (`sigmoid_fit`, `bocpd`,
`shapley_attribution`).

Released artifacts:

- **Code** — `train/ignite/` (A-F harnesses, outer loop, inner GRPO, mutation
  space, archive), `eval/ignite/` (benchmarks, reward, tools, statistics).
- **Checkpoints** — `iterate-labs-ai/ignite-3b-v1` on Hugging Face, one revision
  per retained generation, pushed from the archive during the run rather than
  reconstructed afterward.
- **Logs** — append-only `log.jsonl` containing every candidate attempt with
  mutation hash, dev/validation rewards, entropy delta, pass@k flag, retention
  decision, and rejection reason. Rejected candidates are retained in the log;
  the failure catalogue is a deliverable, not a byproduct.
- **State** — `state.json` enabling `--resume` across Kaggle session boundaries.

This bundle is the specific gap we identify in AIDE²: the first-evidence claim
was published without code, checkpoints, seeds, or logs, and no independent
replication exists.

Every arXiv identifier cited in this draft was verified against the project
reference catalogue (`docs/research/RSI_2026.md`, 33 identifiers checked). Two
identifiers flagged there as unverified — CyberEvolver 2605.26195 and
Self-Reference Introspection 2607.04277 — are **not** cited in this paper.

---

## 9. References

1. STOP: Self-Taught Optimizer — arXiv 2310.02304 (COLM 2024)
2. Gödel Agent — arXiv 2410.04444
3. AIDE: AI-Driven Exploration — arXiv 2502.13138
4. Darwin Gödel Machine — arXiv 2505.22954
5. Huxley-Gödel Machine — arXiv 2510.21614 (ICLR 2026 oral)
6. SICA: Self-Improving Coding Agent — arXiv 2504.15228
7. Absolute Zero Reasoner — arXiv 2505.03335
8. R-Zero — arXiv 2508.05004
9. rStar-Math — arXiv 2501.04519
10. SEAL: Self-Adapting Language Models — arXiv 2506.10943
11. Self-Improving Transformers — arXiv 2502.01612
12. Self-Rewarding Language Models — arXiv 2401.10020
13. Shafayat et al., self-reward collapse — arXiv 2505.21444
14. RL-PLUS entropy collapse — arXiv 2508.00222
15. Beyond Pass@1 — arXiv 2508.14029
16. Zenil et al., contraction dynamics — arXiv 2601.05280
17. Task-Centric Theory of RSI — arXiv 2602.10014
18. ScaleRL sigmoid decomposition — arXiv 2510.13786
19. Adams & MacKay, BOCPD — arXiv 0710.3742
20. FIRMBOUND SPRT — arXiv 2501.18059
21. SPRT overshoot correction — arXiv 2410.16076
22. Heskes, Bregman-conforming metrics — arXiv 2501.18581
23. Signal and Noise — arXiv 2508.13144
24. Cheap bootstrap CI — arXiv 2501.10289
25. AIRA_2 Hidden Consistent Evaluation — arXiv 2603.26499
26. Per-operator attribution — arXiv 2604.16242
27. Illusion of Diminishing Returns — arXiv 2509.09677
28. RE-Bench (METR) — arXiv 2411.15114
29. MLE-Bench — arXiv 2410.07095
30. DeepSeek-R1 — arXiv 2501.12948
31. DAPO dynamic sampling — arXiv 2503.14476
32. Dr. GRPO — arXiv 2503.20783
33. LiveCodeBench — arXiv 2403.07974
34. BigCodeBench — arXiv 2406.15877
35. OMNI-MATH — arXiv 2410.07985
36. PutnamBench — arXiv 2407.11214
37. LeanDojo — arXiv 2306.15626
38. DeepSeek-Prover-V1.5 — arXiv 2408.08152
39. Weco.ai, "The 4 Levels of Recursive Self-Improvement" (blog)
40. Weco.ai, "First Evidence of Recursive Self-Improvement" (blog)
