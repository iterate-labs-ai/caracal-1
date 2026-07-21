# When Your Specialist Isn't: A Base-Controlled Audit of a 3B Cybersecurity Model, and the Compute Wall for Same-Model RSI

*Working draft, 2026-07-20. Every quantitative claim below is measured on the
runs in `docs/s07/results/`; anything not yet obtained is marked "pending" and
never estimated. Raw per-bench JSON is committed alongside this document.*

---

## Abstract

We report a base-controlled evaluation of **Caracal-3B**, a Qwen2.5-Coder-3B
model fine-tuned (adapter `s04`) for cybersecurity, and an honest cost analysis
of applying same-model recursive self-improvement (RSI) to it. Two findings hold
up under scrutiny; a third does not, and we retract it.

1. **The headline "specialist" number was the base model's.** On CyberMetric-500
   the adapter scores 85.8% and the *untouched* Qwen2.5-Coder-3B base scores
   84.2% — a 1.6 pp gap inside the bootstrap CI at n=500. Prior positioning of
   "Caracal-3B in the Llama-3.1-8B band with 3B params" credited the fine-tune
   for a capability the base already had. **We retract that claim.**
2. **On the one benchmark that actually discriminates a cyber specialist
   (CTI-Bench RCM, CVE→CWE), the fine-tune is worse than base**: 39.8% vs 44.4%
   (−4.6 pp, n=500), against a published Foundation-Sec-8B ceiling of 72–75%.
   The specialist gap is real and the adapter widens it.
3. **Same-model RSI is mechanically real but empirically undemonstrated here,
   and the standard configuration does not fit free-tier compute.** The outer
   loop proposes its own mutations (2/2 self-sourced, 0 fallback in the one valid
   generation), but at a measured 202 s/GRPO-step the pre-registered
   8-generation × 8-candidate × 150-step protocol costs ≈540 GPU-hours — 18× a
   Kaggle weekly quota. Two attempted runs were invalidated (one by a mis-routed system
   prompt, one by quota exhaustion); we report them as null, not as evidence.

The contribution is methodological: a demonstration that **a small "specialist"
can be entirely explained by its base model unless a base-controlled,
adequately-powered evaluation is run**, plus a reusable RLVR harness for CVE→CWE
and a cost model that says where same-model RSI can and cannot be tested today.

---

## 1. Why this paper is a correction

Small domain-tuned models are routinely announced with a single headline
benchmark and a parameter-efficiency comparison against a larger generalist. The
Caracal project did exactly this internally: CyberMetric-500 = 85.8%, positioned
against Llama-3.1-8B-Instruct (84.7–85.6%) as "specialist parity at 2.7× fewer
parameters."

The claim has two defects that only a base-controlled run exposes:

- **No base control.** The comparison was adapter-vs-other-models, never
  adapter-vs-its-own-base. When we ran the base, 84.2% of the 85.8% was already
  there.
- **Underpowered side-benchmarks.** The supporting numbers (SecEval, MMLU-sec,
  CWE-pred) were n=20–50, giving ±12–22 pp CIs — statistically unable to
  distinguish adapter from base or from noise.

This paper runs the control and reports what survives.

## 2. Setup

- **Models.** `Caracal` = Qwen2.5-Coder-3B-Instruct + LoRA adapter
  `arturpn/caracal-base-3b-s04`. `base` = the same Qwen checkpoint, no adapter.
- **Harness.** Identical prompt template, parser, and decoding for both; single
  Kaggle T4×2 session each; greedy decoding; `\boxed{}` / last-line answer
  extraction shared across all benches (`eval/s07/benches/_common.py`).
- **Benches.** CyberMetric-500, CTI-Bench (RCM + MCQ), SecQA v1/v2, SecBench,
  MMLU computer-security, SecEval, CyberSOCEval. n reported per cell — never
  hidden.
- **Verifier, not judge.** CWE scoring is exact-match on a normalized CWE ID
  (`normalize_cwe`), with optional hierarchical partial credit on the MITRE CWE
  tree (exact 1.0 / ancestor 0.5 / sibling 0.3). No LLM-judge anywhere — a
  deliberate defense against the self-reward collapse of Shafayat et al.
  (2505.21444).

## 3. Result: adapter vs base, same harness

| bench | Caracal | base | Δ | n |
|---|---:|---:|---:|---:|
| **cti_bench.rcm** (CVE→CWE) | 39.8% | **44.4%** | **−4.6** | 500 |
| **cti_bench.mcq** | 44.2% | **50.6%** | **−6.4** | 500 |
| cybermetric | 85.8% | 84.2% | +1.6 | 500 |
| secqa v1 | 99.1% | 99.1% | 0.0 | 110 |
| secqa v2 | 97.0% | 98.0% | −1.0 | 100 |
| **secbench** | **79.7%** | 74.0% | **+5.7** | 300 |
| mmlu_security | 75.0% | 73.0% | +2.0 | 100 |
| seceval | 42.5% | 40.5% | +2.0 | 200 |
| cybersoceval.malware | 14.0% | 20.0% | −6.0 | 100 |
| cybersoceval.threat_intel | 17.0% | 22.0% | −5.0 | 100 |

**Reading.** The adapter's only gain above plausible noise is SecBench
(+5.7 pp, n=300). On the two benchmarks that specifically test CVE→CWE mapping —
the capability a "cyber specialist" is supposed to have — it is 4.6 and 6.4 pp
*below* its own base. CyberMetric, the original headline, is a wash. CyberSOCEval
regresses. This is not the profile of a successful specialization; it is closer
to mild catastrophic interference on the target skill with incidental gains
elsewhere.

**On the retracted claim.** 84.2% base on CyberMetric-500 places the *base model*
in the same Llama-3.1-8B-Instruct band (84.7–85.6% per Cisco's Foundation-Sec
report, arxiv 2508.01059). The "3B specialist parity" story was always a story
about Qwen2.5-Coder-3B being a strong cyber base, not about the fine-tune.

## 4. Where the real gap is

CTI-Bench RCM is the only benchmark here with published, harness-comparable
specialist baselines:

| model | params | RCM | source |
|---|---:|---:|---|
| Foundation-Sec-8B (card) | 8B | 75.3 | HF card |
| Foundation-Sec-8B (report) | 8B | 72.0 | 2504.21039 |
| Foundation-Sec-8B-Instruct | 8B | 69.2 | 2508.01059 |
| Qwen2.5-7B-Instruct | 7B | 57.2 | 2508.01059 |
| **Qwen2.5-Coder-3B base (ours)** | **3B** | **44.4** | this work |
| **Caracal adapter s04 (ours)** | **3B** | **39.8** | this work |

The specialist frontier at RCM is ~72–75. Our base is at 44.4 and the adapter
drags it to 39.8. The honest target for any future training is therefore not
"improve on 85.8" (saturated, base-driven) but **close the ~28 pp RCM gap from
44.4 toward 72–75**, without the regression the current adapter introduces.

## 5. Same-model RSI: mechanism, cost, and two null runs

We built an RSI harness (`train/ignite/`) in which one 3B checkpoint is both the
inner GRPO solver and the outer mutation proposer, with a purely verifiable
CVE→CWE reward (`eval/ignite/reward.py::cyber_rcm_reward`). The intent is to move
the model off 44.4 toward the specialist frontier via RL rather than supervised
LoRA.

**What is real.** In the one internally-valid generation, the outer loop's
proposals were **2/2 self-sourced, 0 fallback** — the mutations came from the
model, not from a random search stand-in. This is the property that separates
same-model RSI from bootstrap (an external frontier model in the outer loop, as
in AIDE², DGM 2505.22954, HGM 2510.21614, SICA 2504.15228). The training reward
signal on CVE→CWE is also ~5× denser than on the math task we first tried
(reward ≈0.30 vs 0.02–0.09), consistent with CWE mapping being a better-shaped
RL target than olympiad math for a 3B model.

**What is not.** We have **no evidence the loop improves the model.** Two runs:

- *Run 1 (invalid — mis-routed prompt).* Trained 118 min but the mutation table
  had no cyber entry and silently fell back to a math-tutor system prompt; the
  model was optimized for CVE→CWE while being told it was a math tutor. Result
  (`delta=0.0000`, nothing retained) is uninterpretable and discarded.
- *Run 2 (invalid — quota).* Cancelled by weekly GPU-quota exhaustion mid-run.

Both are reported as null. Neither supports nor refutes RSI on cyber.

**The compute wall.** Measured cost is **202 s/GRPO-step** on T4×2 (base 3B, LoRA
r=32, `beta=0`, 4 rollouts). The pre-registered protocol (8 generations × 8
candidates × 150 steps = 9 600 steps) is ≈540 GPU-hours. Kaggle free tier grants
30 GPU-hours/week. **The standard RSI configuration is 18× too large for the
platform it was designed on.** We reduced the per-step generation budget for
CVE→CWE (completion 512→160 tokens; the answer, a CWE ID, is 5.6 tokens; prompts
are p90=169) for a ~3× arithmetic reduction in generated tokens per step — but
this is a token-count reduction, **not a verified wall-clock speedup**, and does
not by itself close the 18× gap. Same-model RSI at this scale needs either a
drastically smaller protocol or paid A100/H100 time.

## 6. Infrastructure hardening (found while measuring)

The audit surfaced five silent scoring bugs — each produced a *wrong number
without crashing*, the failure mode that survives a multi-hour run unnoticed:

- **Hierarchical CWE credit was dead.** Partial credit compared prefixed
  `CWE-1004` against a tree keyed by bare `1004`; ancestor/sibling always scored
  0.0. Now normalized at the parser boundary (`CWEParser._key`).
- **`normalize_cwe` had 3 divergent copies**, two with a regex that rejected
  `"CWE 119"` (space). Consolidated to one source; a correct answer written with
  a space was previously counted wrong on the two core specialist benches.
- **SecBench scored n=0** (schema key mismatch: `answers`/`label` vs
  `option_a`). **CyberSOCEval scored 0.0%** (gold is a multi-answer list, code
  read a scalar). Both fixed and now score; without the fix this paper's table
  would carry two zeros.
- **RCM/MCQ collapse could hide as accuracy.** We report `unparsed_frac` to
  separate "stopped emitting a CWE" from "wrong CWE," and note the majority-class
  floor: always answering CWE-79 (22% of train) scores **27.3%** on dev — the
  threshold below which any RL result is collapse, not learning.

Every fix ships with a regression test (`tests/`, 19 total) verified to fail
against the old broken schema, so the wrong numbers cannot silently return.

## 7. Limitations

- Single-seed, single-session per model; no seed-variance decomposition yet.
- Our harness is not byte-identical to Cisco's / Trend's, so cross-source numbers
  (Foundation-Sec, Primus) are positioning, not a controlled ranking. The
  adapter-vs-base comparison **is** controlled (same harness) and is the only
  Δ we assert strongly.
- SecQA v1/v2 at 97–99% for both models suggests saturation or contamination;
  we report but do not lean on them.
- No RSI improvement result. The mechanism is shown; the effect is not.

## 8. What we claim, precisely

- **Strong (base-controlled):** the Caracal `s04` adapter does not improve, and
  on CVE→CWE actively harms, cyber capability over its own base. The
  CyberMetric "specialist" headline is a base-model property.
- **Measured:** RCM headroom to the 8B specialist frontier is ~28 pp from base;
  same-model RSI costs 202 s/step and ≈540 GPU-h at full protocol.
- **Not claimed:** any evidence that same-model RSI improves the model. Pending
  a valid, adequately-resourced run.

## References (arXiv, verified)

2504.21039 Foundation-Sec-8B · 2508.01059 Foundation-Sec-8B-Instruct ·
2406.07599 CTI-Bench · 2412.20787 SecBench · 2509.20166 CyberSOCEval ·
2505.09388 Qwen3 · 2505.21444 Shafayat (self-reward collapse) ·
2505.22954 Darwin-Gödel Machine · 2510.21614 Huxley-Gödel Machine ·
2504.15228 SICA · 2310.02304 STOP · 2510.13786 ScaleRL ·
0710.3742 Adams-MacKay BOCPD · 2501.18059 FIRMBOUND.

Raw results: `docs/s07/results/{caracal_adapter_s04_gpu,base_qwen25_coder_3b_gpu}.json`.
Verified baseline table: `docs/s07/CYBER_BASELINES_VERIFIED.md`.
