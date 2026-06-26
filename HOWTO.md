# HOWTO · workflow diario

## Apos completar SETUP

### Cada dia

1. Ver [issues](https://github.com/iterate-labs-ai/caracal-1/issues) abertas
2. Comentar `claim` na que voce vai pegar
3. Ver [SCHEDULE.md](SCHEDULE.md): tem slot livre? quer bookar?

### Pra bookar slot Kaggle

1. `git checkout dev && git pull`
2. `git checkout -b book-slot-N`
3. Editar SCHEDULE.md trocando linha por `in-progress · seu_handle · timestamp`
4. PR pequena contra `dev` · self-merge

### Pra rodar treino na sua sessao

1. Abrir https://www.kaggle.com/code · novo notebook
2. Copiar codigo de `train/notebooks/kaggle_continued_pretrain.ipynb`
3. Editar 5 variaveis topo:
   - `SESSION` (1-5)
   - `FOUNDER_HANDLE` (seu username Kaggle)
   - `RESUME_DATASET` (`None` se sessao 1, senao `usuario_anterior/caracal-base-3b-sNN`)
   - `OUTPUT_DATASET_SLUG` (`caracal-base-3b-sNN`)
   - `STEPS` (900 default)
4. Settings: GPU T4 x2 + Internet ON + Persistence
5. Save & Run All
6. ~10h depois dataset publicado automaticamente

### Apos terminar sessao

1. Confirmar dataset em `https://www.kaggle.com/datasets/SEU_USUARIO/caracal-base-3b-sNN`
2. PR pequena: SCHEDULE.md `in-progress` -> `done · dataset_slug`
3. Avisar grupo

### Pra fazer mudanca de codigo

1. `git checkout dev && git pull`
2. `git checkout -b cN-curta-descricao`
3. Commits atomicos
4. `git push -u origin cN-curta-descricao`
5. PR contra `dev` linkando issue: `Closes #N`
6. Pedir review de outro fundador
7. Self-merge apos approval
8. Fechar issue

## Branches

| Branch | Pra que |
|---|---|
| `main` | release estavel (futuro v0 merge) |
| `dev` | branch de trabalho default |
| `tN-work` | pre-criadas por tarefa Sprint 0 |
| `book-slot-N` | bookar slot SCHEDULE.md |
| `cN-fix-Y` | feature branches livres |

## Quando algo quebra

| Situacao | Acao |
|---|---|
| Treino travou (OOM, CUDA error) | Screenshot log + avisa grupo · proximo pega ultimo ckpt salvo (save_every=50) |
| Kaggle Dataset push falhou | Manualmente: `kaggle datasets create -p /kaggle/working/ckpt-out --public` |
| Slot conflict (2 founders booking) | Primeiro PR merged wins, perdedor re-book proximo |
| Bug critico no codigo | Issue urgente + 2 founders debug |
| Kernel timeout Kaggle 12h | Confirmar ultimo ckpt salvo · proximo session retoma |

## Docs importantes

- [SETUP.md](SETUP.md) · primeira vez
- [PROXIMA_SESSAO.md](PROXIMA_SESSAO.md) · quickstart visual pra pegar a proxima sessao do relay
- [SCHEDULE.md](SCHEDULE.md) · slot booking + tabela sessoes
- [CONTRIBUTING.md](CONTRIBUTING.md) · PR style
- [README.md](README.md) · overview tecnico
- [docs/architecture.md](docs/architecture.md) · arquitetura modelo
- [docs/glossary.md](docs/glossary.md) · termos

## Comandos uteis

```bash
# Smoke test local sem treinar
python scripts/smoke_test.py

# Avaliar probe set
python eval/run_probe.py --adapter ./ckpt-out --out probe.json

# Comparar com Qwen base
python eval/compare_baseline.py --adapter ./ckpt-out --out baseline.json

# Check decontamination
python eval/check_decontamination.py --strict
```
