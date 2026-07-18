# Ignite-3B Kaggle Notebook Audit (pre-run)

Date: 2026-07-18
Auditor: static audit, no notebook edited, no GPU spent.
Reference baseline: `notebooks/kaggle/ignite/S03_gen0_2.ipynb` (debugged over 6 Kaggle runs).
Code cross-checked at local HEAD `dee3571`; remote `origin/s07-hybrid-agentic` = `22ec339` (has all C_rsi_outer / inner_grpo fixes; the one unpushed local commit `dee3571` is S01-only and does not affect these notebooks).

## Scope note

All six notebooks `git clone --depth 1 -b s07-hybrid-agentic` the repo, so they inherit whatever is on the remote branch. The remote already carries the T4 loader (`load_model_and_tok` strips `unsloth/` and `-bnb-4bit`), the `free_gpu` VRAM helper, and the GRPO batch-divisibility guard. That means several notebook-level bugs are *masked* at the Python level but still break at the pip level.

---

## The S03 reference contract

Every notebook that trains must reproduce these, from `S03:install`, `S03:hf-login`, `S03:data-load`, `S03:run-C`:

| # | Contract | S03 cell |
|---|----------|----------|
| R1 | `pip uninstall -y torchao` before AND after the install | `install` |
| R2 | `torch==2.6.0` + `torchvision==0.21.0` from `--index-url https://download.pytorch.org/whl/cu124` | `install` |
| R3 | `transformers==4.49.0`, `peft==0.14.0`, `trl==0.15.2` pinned exactly | `install` |
| R4 | `antlr4-python3-runtime==4.11` (sympy `parse_latex` in `eval/ignite/tools/math_verify.py:42`) | `install` |
| R5 | No `unsloth`, no `vllm`, no `FastLanguageModel` | `install` |
| R6 | `BASE_MODEL = 'Qwen/Qwen2.5-3B-Instruct'` | `config` |
| R7 | Import smoke test (`Qwen2ForCausalLM` + torchao absent assert) | `install` |
| R8 | Kaggle secret fetch wrapped in `try/except` | `hf-login` |
| R9 | Build `omni_math_{train,dev,val}.jsonl` in-kernel | `data-load` |
| R10 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the training subprocess | `run-C` |

**None of S04, S05, S07, S08, S09 satisfy R1-R5, R8, R9, R10.** All five carry the identical pre-fix install line.

---

## S04_gen3_5.ipynb — Cond C gen 3-5 (resume)

| Sev | Cell | Finding | Fix |
|-----|------|---------|-----|
| BLOCKER | `config` | `BASE_MODEL = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'` (violates R6) | `BASE_MODEL = 'Qwen/Qwen2.5-3B-Instruct'`. Not fatal at runtime (`C_rsi_outer.py:41` rewrites the string) but it is a lie in the log and breaks if the loader is ever simplified. |
| BLOCKER | `install` | Installs `unsloth>=2025.1.0` and `vllm>=0.6.0`; no torch pin; no torchao uninstall; no `antlr4` pin; `transformers>=4.46.0` unpinned (violates R1-R5) | Replace the whole cell with the three lines from `S03:install` verbatim. unsloth and vllm each drag in their own torch and will clobber the pin; nothing in `train/ignite/` imports either (`inner_grpo.py` sets `use_vllm=False`). |
| BLOCKER | `install` | No torch pin → Kaggle default torch 2.10 → if the kernel lands on a **P100 (SM 6.0)** every CUDA op fails | pin `torch==2.6.0` cu124 |
| BLOCKER | `hf-login` | `login(token=UserSecretsClient().get_secret('HF_TOKEN'))` unguarded (violates R8) | wrap in `try/except Exception` exactly as `S03:hf-login`. Secrets are not injected on API/CLI kernel push — cell 4 kills the run. |
| BLOCKER | `pull-prev` | `PREV_KAGGLE_DATASET = 'vitorscrt/ignite-3b-cond-c-s03'` **does not exist**. S03 has no publish cell at all — its last cell is `summary`, which only prints. | Either add a publish cell to S03 (write `dataset-metadata.json` into `/kaggle/working/cond_C_run` + `kaggle datasets create`), or attach the S03 output as a kernel `dataset_sources` in kernel-metadata and drop the CLI download. |
| BLOCKER | `pull-prev` | `kaggle datasets download` requires `KAGGLE_USERNAME`/`KAGGLE_KEY`. A Kaggle kernel is **not** authenticated to the Kaggle API by default; there is no `kaggle.json` in the container | Prefer attaching the dataset to the kernel (mounted read-only at `/kaggle/input/<slug>`) over the CLI. If the CLI is kept, pull the creds from Kaggle Secrets — which also fails on API push (see R8). |
| BLOCKER | `resume-C` | Datasets `data/ignite/omni_math_{train,dev,val}.jsonl` **do not exist in the fresh kernel**. They are not in git (`git ls-files data/ignite/` returns only `build_*.py`), and `pull-prev` unzips into `OUT_ROOT=/kaggle/working/cond_C_run`, not into `data/ignite/` | Port the `S03:data-load` cell verbatim (it runs `python -m data.ignite.build_omni_math` then slices 200/100/100 with `random.Random(42)`). The seed must stay 42 or the train/dev/val split shifts between sessions and the val contamination gate is void. |
| BLOCKER | `resume-C` | **`--resume` does not restore the evolved adapter.** `C_rsi_outer.py:101` sets `v_adapter = v0_adapter` unconditionally. `arch.save_state` writes `{"v_adapter": ...}` at line 186 but `load_state()` at line 98 is only read for `last_gen`. So resuming at gen 3 restarts from the **base model** while skipping gens 0-2. Line 105 then measures `r_prev` on the base too, so `delta_pp` is computed against the wrong reference. | Fix in `C_rsi_outer.py`: after line 98, `v_adapter = state.get("v_adapter") or v0_adapter` (treat `""` as None). Do not run S04/S05 before this lands — the session would burn 12h producing a run that silently is not a continuation. |
| WARN | `config` | `GENS_TOTAL = 6` assumes S03 finished gen 2. S03 actually ships `GENS=1, CANDS=3, STEPS=50` → only gen 0 exists. Resume computes `start_gen = last_gen + 1 = 1`, so this session runs gens **1-5**, not 3-5 | Reconcile the plan: either raise S03's GENS or retitle S04. Also `CANDS=8, STEPS=150` × 5 gens ≈ 40 GRPO runs of 150 steps plus 40 dev evals + 10 val evals — will not fit a 12h T4 session. Budget from S03's measured throughput first. |
| WARN | `resume-C` | No `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (violates R10) | copy the `env` dict from `S03:run-C` and pass `env=env` |
| WARN | `publish` | `kaggle datasets version -p` runs before the dataset exists; the `create` fallback is only reached on non-zero exit — but `create` reads the same `dataset-metadata.json` whose `id` is `vitorscrt/...`, which must match the authenticated account or 403s | Verify the Kaggle account slug matches `vitorscrt`; the HF org is `iterate-labs-ai`, so these are two different identities. |
| WARN | `publish` | `--public` uploads the entire `cond_C_run` tree, i.e. every LoRA candidate adapter (8 cands × N gens), not just `log.jsonl` + `state.json` | Publish only `log.jsonl`, `state.json`, and the retained `v_k` dirs. |
| WARN | `clone` | No `git rev-parse HEAD` print (S03 has it) | keep provenance |

---

## S05_gen6_8.ipynb — Cond C gen 6-8 final

Inherits every S04 finding. Additional:

| Sev | Cell | Finding | Fix |
|-----|------|---------|-----|
| BLOCKER | `config` | `BASE_MODEL = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'` (R6) | `Qwen/Qwen2.5-3B-Instruct` |
| BLOCKER | `install` | Identical pre-fix install line (R1-R5) | replace with `S03:install` |
| BLOCKER | `clone` | Unguarded `UserSecretsClient().get_secret('HF_TOKEN')`, and it is fused into the clone cell so the clone also dies (R8) | split the cell, wrap the secret |
| BLOCKER | `pull-prev` | `PREV_KAGGLE = 'vitorscrt/ignite-3b-cond-c-s04'` does not exist — depends on S04's `publish` cell, which itself is unverified | see S04 |
| BLOCKER | `resume-C-final` | Datasets not built (R9) | port `S03:data-load` |
| BLOCKER | `resume-C-final` | Same `--resume` adapter-loss bug (`C_rsi_outer.py:101`) | fix upstream first |
| BLOCKER | `ship-final` | `last['v_ckpt']` is an **absolute path from a previous kernel** (`/kaggle/working/cond_C_run/genN/candM`). It only resolves if the pulled Kaggle dataset unzipped with that exact layout into `OUT_ROOT`. If `pull-prev` flattened or nested it, `upload_folder` raises | Resolve defensively: `Path(OUT_ROOT) / Path(last['v_ckpt']).relative_to(OUT_ROOT)` with an `exists()` assert before upload. |
| WARN | `ship-final` | `gens[-1]` takes the last `type == 'generation'` row across the **appended** log. If `pull-prev --force` did not overwrite `log.jsonl`, or if a rerun appended, this can pick a stale/rejected entry | assert `last['gen'] == GENS_TOTAL - 1` before shipping |
| WARN | `ship-final` | Uploads to `iterate-labs-ai/ignite-3b-v1` at repo **root** (no `path_in_repo`), whereas `archive.push_hf` (`archive.py:87`) pushes intermediates under `v{gen}/`. Two different layouts in the same project | Intentional or not, S09 depends on the root layout — document it. |
| WARN | — | No Kaggle publish cell, but S10 expects `vitorscrt/ignite-3b-cond-c-s05` | add a publish cell mirroring `S04:publish`, else S10 has no input |

---

## S07_ablation_D_E.ipynb — Cond D (LLM-judge) + Cond E (frozen)

| Sev | Cell | Finding | Fix |
|-----|------|---------|-----|
| BLOCKER | `config` | `BASE = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'` (R6) | `Qwen/Qwen2.5-3B-Instruct` |
| BLOCKER | `install-clone` | Identical pre-fix install (R1-R5), plus unguarded secret (R8), all fused into one cell | split into `install` / `clone` / `hf-login` and copy S03's versions |
| BLOCKER | `cond-D` / `cond-E` | Datasets not built (R9) | port `S03:data-load` |
| BLOCKER | `cond-D` | **Cond D is not implemented.** `D_llmjudge.py:33` calls `outer_loop(bench="llmjudge")`; that string reaches `inner_grpo.train_lora` → `eval/ignite/reward.py:82` `raise ValueError(f"unknown bench: llmjudge")`. `C_rsi_outer.py:134` catches `(RuntimeError, ValueError)`, records `train_failed`, and continues — so **all 4 candidates × 4 gens fail silently**, `cands_out` is empty, and the run prints "no candidates trained, skip" 4 times and exits 0. With `check=False` the notebook reports success and produces an empty `log.jsonl`. | Implement an `llmjudge` branch in `build_reward_fn` (self-scoring via `eval.s07.benches._common.generate`) before running S07. Additionally: even if the reward existed, evaluation still uses `BENCH_REGISTRY['omni_math']` (`C_rsi_outer.py:63`), i.e. RLVR — so the ablation would measure the wrong thing. Cond D needs a judge-based eval path too. |
| BLOCKER | `cond-E` | **`--steps 0` does not freeze the inner loop.** It flows to `GRPOConfig(max_steps=0)` (`inner_grpo.py:128`). HF `Trainer` treats `max_steps > 0` as the override; `0` falls through to `num_train_epochs=1`, so Cond E trains a **full epoch** over the 200-row train set. The "frozen inner" ablation is invalid. | Add an early return in `train_lora`: `if steps <= 0: save the incoming adapter (or an identity LoRA) to out_dir and return out_dir`. Only the mutation scaffolding should vary in Cond E. |
| WARN | `cond-E` | `save_steps = max(1, steps // 3)` = **1** when `steps=0` (`inner_grpo.py:136`) → a checkpoint written every single step | same fix as above |
| WARN | `cond-D` / `cond-E` | Both use `check=False`, so a failing subprocess is invisible; combined with the Cond D silent-empty path, the notebook can "pass" with zero data | use `check=True`, or assert `log.jsonl` has ≥1 `type == 'generation'` row after each run |
| WARN | `config` | `OUTPUT_DATASET = 'vitorscrt/ignite-3b-ablations-de'` is defined and **never used** — there is no publish cell. Results live only in `/kaggle/working` | add a publish cell, or the 12h session is unrecoverable |
| WARN | `cond-D` / `cond-E` | No `PYTORCH_CUDA_ALLOC_CONF` (R10); two sequential 3B RSI runs in one kernel makes fragmentation worse than S03, not better | set the env on both subprocesses |
| WARN | — | Two full ablations (4 gens × 4 cands × 100 steps each = 32 GRPO runs + 40 evals) in one 12h session is not credible given S03 needed to be cut to 1 gen / 3 cands / 50 steps | split into two sessions or cut to 2 gens |
| OK | `cond-D` | CLI flags match `D_llmjudge.py` argparse: `--base --dataset-train --dataset-dev --dataset-val --bench-name --gens --cands --steps --out`. Correctly does **not** pass `--bench` (D has no such flag) | — |
| OK | `cond-E` | Flags match `E_frozen.py` argparse, including `--bench math`. Correctly does not pass `--v0-adapter` (E has no such flag) | — |

---

## S08_cond_F_7b.ipynb — Cond F, Qwen2.5-7B

| Sev | Cell | Finding | Fix |
|-----|------|---------|-----|
| BLOCKER | `config` + `cond-F` | **7B will not fit.** `BASE_7B` is rewritten to `Qwen/Qwen2.5-7B-Instruct` (`C_rsi_outer.py:41`) and loaded `torch_dtype=torch.float16, device_map={"": "cuda:0"}` (`C_rsi_outer.py:45-50`, `inner_grpo.py:79-84`). 7.6B params × 2 bytes ≈ **15.2 GB** of weights on a 15.9 GB T4 — before LoRA, gradients, KV cache, and GRPO's `num_generations=4` rollouts at `max_completion_length=512`. Hard-pinning `cuda:0` means the second T4 is unused. OOM on the very first `from_pretrained`. | Three options, pick one before spending quota: (a) run Cond F on a P100-free A100/L4 elsewhere; (b) 4-bit `BitsAndBytesConfig` load + QLoRA, which changes the condition semantics vs Cond C's fp16 and must be disclosed in the writeup; (c) `device_map="auto"` to shard across both T4s — but note `free_gpu` (`C_rsi_outer.py:24`) and the `.to("cuda:0")` assumptions elsewhere are single-device, so this needs testing. Do **not** launch as written. |
| BLOCKER | `install-clone` | Identical pre-fix install (R1-R5) + unguarded secret (R8) | copy `S03:install` + `S03:hf-login` |
| BLOCKER | `cond-F` | Datasets not built (R9) | port `S03:data-load` |
| WARN | `config` | `HF_REPO = 'iterate-labs-ai/ignite-7b-cond-f'` is defined but never used — `F_7b.py` has **no `--hf-repo` flag** (`F_7b.py:13-26` argparse) and `outer_loop` is called without `hf_repo` (`F_7b.py:28-40`). Nothing is pushed anywhere and there is no Kaggle publish cell either | Add `--hf-repo` to `F_7b.py` argparse and forward it, or add a publish cell. As written, a successful 12h run leaves zero durable artifact. |
| WARN | `cond-F` | No `PYTORCH_CUDA_ALLOC_CONF` (R10) — most needed here of all notebooks | set it |
| OK | `cond-F` | CLI flags all exist in `F_7b.py` argparse | — |

---

## S09_ignition_test.ipynb — L2 ignition gate

| Sev | Cell | Finding | Fix |
|-----|------|---------|-----|
| BLOCKER | `config` | `VN_REPO = 'iterate-labs-ai/ignite-3b-v1'` **does not exist yet** — it is created by `S05:ship-final`. `snapshot_download` in `ignition-run` raises `RepositoryNotFoundError` | gate on S05 completing |
| BLOCKER | `config` | `BASE = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'` (R6) | `Qwen/Qwen2.5-3B-Instruct` |
| BLOCKER | `install-clone` | Identical pre-fix install (R1-R5) + unguarded secret (R8) | copy S03 cells |
| BLOCKER | `ignition-run` | Datasets not built (R9) | port `S03:data-load` — and note this notebook **must** reuse S03's `random.Random(42)` split or the held-out slice differs from the one v_N was selected on, which invalidates the comparison |
| BLOCKER | `ignition-run` | **The experiment is confounded.** `--v0-adapter vn_adapter` sets `v_adapter` in `outer_loop`, which is used for *two* things: as the proposer (`C_rsi_outer.py:104,112`) **and** as `adapter_in` for every candidate's training init (`C_rsi_outer.py:126`). So the v_N arm starts its candidates from v_N's *inner* weights while the v_0 arm starts from base. Any measured gain is inner-model quality, not outer-proposer quality — which is exactly the claim S09 is supposed to isolate. | Split the parameter: add `--proposer-adapter` (used only for `propose_via_self`) distinct from `--v0-adapter` (used only as `adapter_in`). Both arms must train candidates from the **same** init. Without this the L2 result is not defensible. |
| BLOCKER | `mcnemar` | **The notebook does not compute McNemar.** `mcnemar_test` is imported and never called; the cell prints `[TODO] Run eval.ignite.run_all_ignite --adapter both, then mcnemar_test(per_sample)`. The stated success criterion (p<0.01 + sigmoid asymptote CI>0) is not produced | Implement: re-eval both winning adapters on the identical val slice with per-sample correctness vectors, then `mcnemar_test(a_correct, b_correct)` (`eval/ignite/stats.py:151`). `BENCH_REGISTRY['omni_math']` currently returns only aggregate `accuracy` (`C_rsi_outer.py:65`) — a per-sample output path has to be added first. |
| WARN | `mcnemar` | `best_val_per_sample` returns `[]` (a list) on empty input, then `v0_best.get("val_r")` → `AttributeError: 'list' object has no attribute 'get'` | return `None` and guard, or return `{}` |
| WARN | `mcnemar` | `mcnemar_test` needs `statsmodels` (`stats.py:152`); the `install-clone` cell does not install it | Kaggle images ship statsmodels, but pin it explicitly since the S03 install rewrites the torch stack |
| WARN | `ignition-run` | Two full 1-gen × 8-cand × 100-step runs sequentially in one session | budget or reduce CANDS |
| WARN | `ignition-run` | No `PYTORCH_CUDA_ALLOC_CONF` (R10) | set it |
| WARN | `ignition-run` | `snapshot_download(repo_id=VN_REPO)` assumes adapter files at repo **root**. That holds only because `S05:ship-final` uploads without `path_in_repo`; `archive.push_hf` (`archive.py:87`) uses `v{gen}/` for the intermediate repo. If the source is ever switched to `ignite-3b-cond-c`, `PeftModel.from_pretrained` will fail on the missing root `adapter_config.json` | assert `Path(vn_adapter, 'adapter_config.json').exists()` after download |
| WARN | — | No publish/output cell | add one |

---

## S10_shapley_writeup.ipynb — analysis

Analysis-only: no model load, so R1-R7 do not bite. Its install cell (`huggingface_hub scipy numpy matplotlib seaborn kaggle`) is fine as-is.

| Sev | Cell | Finding | Fix |
|-----|------|---------|-----|
| BLOCKER | `load-archive` | `ARCHIVE_DATASET = 'vitorscrt/ignite-3b-cond-c-s05'` **does not exist** — S05 has no Kaggle publish cell (it only pushes to HF). `check=True` → the notebook dies on cell 3 | add a publish cell to S05, or point at whatever S04/S05 actually produce |
| BLOCKER | `load-archive` | Same unauthenticated `kaggle datasets download` problem as S04 | attach as a kernel dataset source instead |
| WARN | `shapley` | `shapley_attribution` (`stats.py:170`) only sums `val_r` over rows with `retained` truthy. `record_candidate` only sets `retained` on the per-generation winner row, written with `cand_id=-1` (`C_rsi_outer.py:164-174`). So "top-10 mutations" can have at most **one row per generation** — with 9 gens the top-10 is really a top-≤9, and rejected mutations get zero credit. Despite the name, this is a running sum, not a Shapley value | Either rename to `cumulative_retained_val_r`, or compute real marginal contributions over the full candidate set (all `dev_r` rows are available). Do not describe the output as "Shapley" in the paper as-is. |
| WARN | `reward-hacking` | `hack = 1 if dev > val + 0.05` uses `c.get('dev_r', 0)` / `c.get('val_r', 0)`. Every non-winner candidate has `val_r = None` (`archive.py:36` default), so `c.get('val_r', 0)` returns **`None`, not 0** — the `dev > val + 0.05` comparison then raises `TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'`. Saved only by the `val_r is not None` filter one line above, which restricts the analysis to ~1 candidate per generation | Make the fallback explicit (`or 0.0`) and note the sample size: with one row per gen this rate is 0% or 100% per generation and statistically meaningless. To measure hacking properly, val-eval all candidates, not just top-2. |
| WARN | `sigmoid-plot` | Uses `log` defined in the `reward-hacking` cell — hidden inter-cell dependency; out-of-order execution gives `NameError` | re-read `log.jsonl` in the cell |
| WARN | `sigmoid-plot` | `compute = [i+1 for i in range(len(val_r))]` uses generation index as the compute proxy, but generations have unequal compute (S03 = 3 cands × 50 steps, S04/S05 = 8 × 150). The sigmoid asymptote fit — a stated L2 criterion in S09 — is fit against a mis-scaled x-axis | use cumulative GRPO steps (`cands × steps` accumulated per gen) as x |
| WARN | `sigmoid-plot` | Guarded by `if gens:` — silently produces nothing when the log has no accepted generation, which is the likely outcome if S07's Cond D log is ever fed in | `raise` instead |
| WARN | `install-clone` | `seaborn` installed, never imported | drop |
| OK | `shapley` / `sigmoid-plot` | `shapley_attribution`, `sigmoid_fit`, `bocpd` all exist in `eval/ignite/stats.py` with matching signatures; returned keys `mutation_hash` / `cumulative_contribution` match the print loop | — |

---

## Prioritized fix list

Fix in this order. Items 1-3 are code fixes in `train/ignite/` and block multiple notebooks each; 4-8 are per-notebook.

| # | Fix | Where | Unblocks |
|---|-----|-------|----------|
| 1 | **Restore `v_adapter` on resume.** `C_rsi_outer.py:98-101` — read `state.get("v_adapter")` and use it when it is a non-empty string. Without this, every `--resume` session silently restarts from base | `train/ignite/C_rsi_outer.py` | S04, S05 |
| 2 | **Port the S03 install + secrets + data-load cells** into S04, S05, S07, S08, S09 verbatim (R1-R5, R8, R9, R10). This is a mechanical copy of three cells and removes ~25 of the findings above | 5 notebooks | all training notebooks |
| 3 | **`steps <= 0` must actually skip training.** `inner_grpo.train_lora` early-return before `GRPOConfig`. Currently `max_steps=0` silently trains one full epoch | `train/ignite/inner_grpo.py:124` | S07 Cond E |
| 4 | **Implement the `llmjudge` reward** (and a judge-based eval path) or delete Cond D. As written it produces an empty run that exits 0 | `eval/ignite/reward.py:52-82` | S07 Cond D |
| 5 | **Decide Cond F's hardware.** 7B fp16 cannot load on one T4. Choose QLoRA-4bit (and disclose the asymmetry vs Cond C) or move off Kaggle. Also add `--hf-repo` to `F_7b.py` so the result survives | `train/ignite/F_7b.py`, S08 | S08 |
| 6 | **Add publish cells** to S03 (→ `*-s03`), S05 (→ `*-s05`), S07, S08. Four notebooks currently write results only to ephemeral `/kaggle/working` while the next notebook downloads a slug that was never created | S03, S05, S07, S08 | S04, S09, S10 |
| 7 | **Decouple proposer from candidate-init** in `outer_loop` (add `--proposer-adapter`). Until then S09's L2 result is confounded and not publishable | `train/ignite/C_rsi_outer.py:104,126` | S09 |
| 8 | **Implement the McNemar cell** in S09 (per-sample correctness from the bench, then `mcnemar_test`), and fix the `[]`-vs-`.get` crash | S09 `mcnemar`, `eval/ignite/benches/omni_math.py` | S09 |
| 9 | Replace the Kaggle CLI downloads with kernel `dataset_sources` mounts at `/kaggle/input/<slug>` — the CLI is unauthenticated inside a kernel | S04, S05, S10 | S04, S05, S10 |
| 10 | Fix S10's `None`-vs-`0` fallback and rename "Shapley" to what it computes; use cumulative steps as the sigmoid x-axis | S10 | writeup correctness |
| 11 | Re-budget every session. S03 had to be cut to 1 gen / 3 cands / 50 steps to fit 12h; S04-S09 still carry the original 8 cands × 150 steps × 4-6 gens | all | quota |

---

## Dependency order — what can run today

| Notebook | Upstream artifact needed | Exists? | Status |
|----------|--------------------------|---------|--------|
| S03 | none (builds its own data) | — | **RUNNABLE TODAY.** Already debugged. Add a Kaggle publish cell before running, or S04/S10 stay blocked. |
| S07 | none (self-contained ablations) | — | **RUNNABLE after fixes 2, 3, 4.** No upstream artifact dependency — this is the best candidate for the next session once Cond D's reward exists. Run Cond D and Cond E as two separate sessions. |
| S08 | none | — | **BLOCKED on hardware, not artifacts.** 7B fp16 OOMs on T4. Needs fix 5 first. Independent of S03/S04/S05. |
| S04 | Kaggle `vitorscrt/ignite-3b-cond-c-s03` | **NO** — S03 has no publish cell | BLOCKED on S03 + fix 1 + fix 6 |
| S05 | Kaggle `vitorscrt/ignite-3b-cond-c-s04` | **NO** — depends on S04's untested publish cell | BLOCKED on S04 |
| S09 | HF `iterate-labs-ai/ignite-3b-v1` | **NO** — created by `S05:ship-final` | BLOCKED on S05 + fix 7 + fix 8 |
| S10 | Kaggle `vitorscrt/ignite-3b-cond-c-s05` | **NO** — S05 has no publish cell | BLOCKED on S05 + fix 6 |

Critical path: `S03 (+publish) → S04 → S05 → {S09, S10}`. Four sessions serialized, each currently blocked by the missing publish cell and by the `--resume` adapter bug (fix 1), which would corrupt the chain at the first resume. S07 and S08 hang off the side and can be worked in parallel once fixes 3/4 and 5 land respectively.

Recommended next GPU session: **S07 Cond E only** (after fixes 2 and 3) — self-contained, no upstream artifact, and it validates the ported S03 install stack on a second notebook before the serialized C chain commits 4 sessions to it.
