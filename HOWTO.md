# HOWTO · workflow diario

## Apos completar SETUP

### Cada manha

1. Olhar [issues](https://github.com/iterate-labs-ai/caracal-1/issues) abertas com label `sprint-0`
2. Comentar `claim` na que voce vai pegar
3. Olhar [SCHEDULE.md](SCHEDULE.md): tem slot livre? quer bookar?

### Pra bookar slot Kaggle

1. `git checkout -b book-slot-N dev`
2. Editar SCHEDULE.md trocando `pending` por `in-progress · seu_handle · timestamp`
3. PR pequena contra dev
4. Self-merge

### Pra rodar treino (na sua sessao)

1. Abrir https://www.kaggle.com/code · novo notebook
2. Copiar codigo de `train/notebooks/kaggle_continued_pretrain.ipynb`
3. Editar 5 variaveis topo:
   - SESSION (1-5)
   - RESUME_DATASET (None se sessao 1, senao 'usuario_anterior/caracal-base-stepXXXX')
   - OUTPUT_DATASET_SLUG ('caracal-base-stepYYYY')
   - STEPS (4000 default)
   - FOUNDER_HANDLE (seu username Kaggle)
4. Run all
5. ~5 horas depois publica Kaggle Dataset publico

### Apos terminar sessao

1. Confirmar dataset em https://www.kaggle.com/datasets/SEU_USUARIO/caracal-base-stepXXXX
2. PR pequena editando SCHEDULE.md: trocar `in-progress` por `done · dataset USUARIO/SLUG`
3. Sinalizar grupo do time

### Pra fazer mudanca de codigo

1. `git checkout -b cN-curta-descricao dev`
2. Commits atomicos
3. Push: `git push -u origin cN-curta-descricao`
4. PR contra dev linkando issue: `Closes #N`
5. Pedir review de outro fundador
6. Apos merge: `done` na issue, fechar, claim proxima

## Branches

| Branch | Pra que |
|---|---|
| `main` | release estavel (futuro) |
| `dev` | branch de trabalho default |
| `t1-work` a `t10-work` | pre-criadas por tarefa |
| `book-slot-N` | bookar slot SCHEDULE.md |
| `cN-fix-Y` | feature branches |

## Quando algo quebra

| Situacao | Acao |
|---|---|
| Treino travou | screenshot + sinaliza grupo |
| Kaggle Dataset push falhou | log + sinaliza grupo |
| Slot conflict | primeiro book wins, perdedor re-book proximo |
| Bug codigo critico | issue urgente + 2 founders debug |
| Sessao Kaggle morreu | confirmar ultimo checkpoint salvo · proximo retoma da save mais recente |

## Docs importantes

- [SETUP.md](SETUP.md) · primeira vez
- [SCHEDULE.md](SCHEDULE.md) · slot booking
- [CONTRIBUTING.md](CONTRIBUTING.md) · PR style
- [docs/plan/iterate_labs.html](docs/plan/iterate_labs.html) · plano MVP completo
- [docs/architecture.md](docs/architecture.md) · arquitetura modelo
- [docs/glossary.md](docs/glossary.md) · termos por extenso
