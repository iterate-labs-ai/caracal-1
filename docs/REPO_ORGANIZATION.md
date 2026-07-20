# REPO_ORGANIZATION.md

Auditoria 2026-07-19. Nenhum arquivo movido ou deletado. Todos os `git mv`
abaixo sao **DO-NOT-EXECUTE-WHILE-KAGGLE-RUNS**: os notebooks fazem `git clone`
deste repo e tem paths hardcoded.

## Layout atual (topo)

```
caracal-1/
├── README.md HOWTO.md SETUP.md SCHEDULE.md PROXIMA_SESSAO.md   # relay founders (era s07)
├── CONTRIBUTING.md LICENSE reproduce.sh requirements.txt pyproject.toml
├── train/        continued_pretrain.py + s07/(00-07) + ignite/(A-F) + configs/ + notebooks/(LEGACY)
├── eval/         run_*.py (LEGACY s01) + _common.py + s07/ + ignite/ + reports/(vazio)
├── data/         corpus_manifest.yaml + s07/ + ignite/ + decontamination/
├── notebooks/kaggle/  s07/(3 nb) + ignite/(S01-S10)   # notebooks ATIVOS
├── docs/         4 root .md + s07/(7) + research/(5) + plan/(html+pdf)
├── tests/        test_kernel_integrity.py + test_rewards.py
├── harness/ modules/ infra/ archive/   # scaffolds harness v1, quase so README/.gitkeep
├── scripts/      publish_kaggle_dataset.sh smoke_test.py
└── .github/ .ruff_cache/
```

Fora de lugar: `.aislop/session.jsonl` git-tracked; `train/notebooks/` duplica
`notebooks/kaggle/`; eval-root `run_*.py` predata `eval/s07/benches/`;
`train/ignite/archive.py` colide de nome com o dir topo `archive/`.

## Status por path

| Path | Projeto | Status | Motivo |
|---|---|---|---|
| eval/s07/, eval/ignite/, data/*, train/s07/, train/ignite/ | ambos | keep | codigo ativo |
| notebooks/kaggle/s07, .../ignite | ambos | keep | Kaggle clona estes |
| eval/run_*.py, eval/_common.py, eval/compare_baseline.py, eval/shadow_loop.py, probe_set.jsonl | s07 legacy | archive | superado por eval/s07/benches + bench_runner |
| eval/run_bench_all.py | legacy | archive | superado por eval/s07/run_all_benches.py |
| train/notebooks/ (11 kaggle_*.ipynb) | s07 legacy | archive | superado por notebooks/kaggle/s07 |
| train/continued_pretrain.py | s07 | keep-review | referenciado por kaggle_continued_pretrain nb |
| eval/s07/bench_runner.py normalize_cwe (L27), eval/s07/hier_reward.py normalize_cwe (L14) | s07 | merge | 3a copia de _common.normalize_cwe |
| eval/ignite/reward.py cyber_rcm_reward | ignite | flag | reward RSI importa arvore/parser do s07 (ponte fase 2) |
| archive/ (dgm_sampler, map_elites, schema.py) | v1 | keep-scaffold | README diz "codigo pos v1" |
| harness/, modules/, infra/ | v1 | archive-candidate | quase so README+.gitkeep; so stop_pattern_detector.py tem teste |
| .aislop/session.jsonl | — | delete-candidate | log de agente, por no .gitignore |

## Split dois-projetos — NAO limpo

Imports cross-project (documentar, nao quebrar):
- `eval/ignite/benches/_common.py` re-exporta de `eval/s07/benches/_common.py`.
- `train/ignite/mutations.py:154` importa `eval.s07.benches._common.generate`.
- `eval/ignite/reward.py` (cyber_rcm_reward) importa `eval.s07.cwe_tree_parser`,
  `eval.s07.benches._common.normalize_cwe`, `eval.s07.hier_reward.hier_cwe_reward`.
  O reward RSI carrega logica cyber do s07 (ponte "fase 2 cyber"). Extrair os
  helpers compartilhados pra um `eval/common/` neutro depois, pra ignite nao
  depender de s07.

## Morto / stale

- `train/s07/` pula `02_*` (00,01,03-07) — verificar se falta arquivo.
- `docs/s07/HANDOFF.md` lista notebooks que nao existem — plano stale.
- `archive/primitive_library/.gitkeep` vazio; `.ruff_cache/` 4 versoes velhas.

## Naming inconsistente

- `S01`-`S10` (ignite, maiuscula) vs `s07_bench_*`/`s07A` (s07, minuscula/mista)
  vs `train/notebooks/kaggle_tpu_bench_s01.ipynb`.
- ignite conds: arquivos `A_baseline..F_7b.py` vs notebook `S08_cond_F_7b`.
- Escolher um esquema (recomendo `sNN_`, condicao `condX`).

## requirements.txt vs imports

- `sentence-transformers` listado 2x (L27, L46).
- Listado mas zero import: `z3-solver`, `modal`, `qdrant-client`, `scrapling`,
  `beautifulsoup4`, `lxml`, `vllm`, `wandb` (deps scaffold v1). `weco`,
  `latex2sympy2` usados indireto (CLI/transitivo) — manter com comentario.
- Todos os deps usados presentes. Nenhum import faltando.

## Consolidacao docs (proposta)

Merge:
- `SCHEDULE.md` + `PROXIMA_SESSAO.md` + `docs/s07/HANDOFF.md` → **docs/s07/RELAY.md**.
- `docs/s07/BENCH_LEADERBOARD.md` + `BENCH_TARGETS.md` → **docs/s07/BENCHMARKS.md**,
  com `CYBER_BASELINES_VERIFIED.md` (jul 19) como tabela autoritativa.
- `docs/research/RSI_2026.md` + `RSI_STRATEGY.md` → dobrar em **RSI_PROOF_PLAN.md**;
  manter `PAPER_DRAFT.md` e `NOTEBOOK_AUDIT.md` standalone.

Manter: README, HOWTO, SETUP, CONTRIBUTING, docs/{architecture,glossary,
learning_trail}.md, docs/s07/{KAGGLE_ADAPTATION,NEXT_TRAINING_ROUND,PAPERS}.md,
docs/plan/.

## Moves EXECUTADOS 2026-07-20

Feitos com todos os kernels parados (quota GPU esgotada = nada clonando o repo).
Verificado antes: nenhum notebook em `notebooks/kaggle/` referencia esses
arquivos, e nada fora de `eval/s07` / `eval/ignite` importava `eval/_common.py`.

- `eval/run_*.py` (9), `eval/_common.py`, `eval/shadow_loop.py`,
  `eval/compare_baseline.py`, `eval/probe_set.jsonl`, `eval/held_out_2026.yaml`
  → `archive/legacy_eval_s01/`
- `train/notebooks/` → `archive/legacy_notebooks_s01/`
- `.aislop/session.jsonl` untracked + `.aislop/` no `.gitignore`

Mantidos em `eval/`: `__init__.py` (faz o pacote pros imports `eval.s07.*` /
`eval.ignite.*`) e `check_decontamination.py` (ainda em uso).

Verificação pós-move: 13 testes passando e os 8 módulos que os notebooks
importam (`eval.s07.benches`, `eval.ignite.benches`, `eval.ignite.reward`,
`train.ignite.{C_rsi_outer,inner_grpo,mutations}`, `data.ignite.build_cyber_rcm`,
`eval.check_decontamination`) todos resolvem.

## Moves recomendados (histórico) — DO-NOT-EXECUTE-WHILE-KAGGLE-RUNS

```bash
# Rodar SO quando nenhum notebook Kaggle estiver clonando este repo.
git mv eval/run_bench_all.py eval/run_cti_bench.py eval/run_cybermetric.py \
       eval/run_secqa.py eval/run_mmlu_security.py eval/run_humaneval.py \
       eval/run_probe.py eval/run_cybergym.py eval/run_cybergym_local.py \
       eval/run_hosted.py eval/shadow_loop.py eval/compare_baseline.py \
       eval/_common.py eval/probe_set.jsonl eval/held_out_2026.yaml  archive/legacy_eval_s01/
git mv train/notebooks archive/legacy_notebooks_s01
git rm --cached .aislop/session.jsonl   # depois add `.aislop/` no .gitignore
# Depois de mover eval/_common.py, atualizar imports em eval/check_decontamination.py.
```

## Testes faltando — por risco

Antes desta sessao so `tests/test_kernel_integrity.py`. Adicionado
`tests/test_rewards.py` (item 5 abaixo, ja cobre hier_cwe_reward + cyber_rcm).
Restante em ordem de prioridade:

1. `eval/s07/benches/secbench.py::eval_secbench` — key option_a mismatch → n=0.
2. `eval/s07/benches/cybersoceval.py::eval_cybersoceval` — parse vazio → 0.0.
3. `normalize_cwe` (3 copias: `_common.py:15`, `hier_reward.py:14`,
   `bench_runner.py:27`) — consolidar e testar 1x.
4. `eval/s07/benches/cti_bench.py` RCM scoring.
5. ~~`eval/ignite/reward.py`: math/code/cyber_rcm~~ FEITO (test_rewards.py).
6. Stats gates `eval/ignite/stats.py`: mcnemar, sprt_firmbound, sigmoid_fit.
7. RSI outer loop retention/keep-discard (smoke).
8. Row-count guards por bench loader.
