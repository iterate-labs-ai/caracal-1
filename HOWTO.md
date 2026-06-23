# HOWTO · workflow diario

## Apos completar SETUP

### Cada manha

1. Olhar [issues](https://github.com/iterate-labs-ai/caracal-1/issues) abertas com label `sprint-1`
2. Comentar `claim` na que voce vai pegar
3. Olhar [SCHEDULE.md](SCHEDULE.md): tem slot livre? quer bookar?

### Pra bookar slot Kaggle

1. `git checkout -b book-slot-N dev`
2. Editar SCHEDULE.md trocando `pending` por `in-progress · seu_handle · timestamp`
3. PR pequena contra dev
4. Self-merge

### Pra rodar treino (na sua sessao)

1. Abrir [Kaggle](https://www.kaggle.com) novo notebook
2. Importar `train/notebooks/kaggle_continued_pretrain.ipynb`
3. Editar variaveis topo (SESSION_NUMBER, RESUME_REVISION, OUTPUT_REVISION)
4. Configurar Kaggle Secrets: HF_TOKEN + WANDB_API_KEY (so primeira vez)
5. Run all
6. Sair · ~9h depois push automatico via SIGTERM handler

### Apos terminar sessao

1. Confirmar revision no HF Hub: <https://huggingface.co/iterate-labs/caracal-base-3b-v0>
2. PR pequena editando SCHEDULE.md: trocar `in-progress` por `done · revision step-XXXX`
3. Sinalizar no grupo do time

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
| `book-slot-N` | bookar slot SCHEDULE.md |
| `cN-fix-Y` | feature branches |

## Quando algo quebra

| Situacao | Acao |
|---|---|
| Treino travou | screenshot + sinaliza grupo |
| HF Hub push falhou | log + sinaliza grupo |
| Slot conflict | primeiro book wins, perdedor re-book proximo |
| Bug codigo critico | issue urgente + 2 founders debug |
| Sessao Kaggle morreu | SIGTERM handler ja salvou. Confirmar revision no HF Hub.|

## Docs importantes

- [SETUP.md](SETUP.md) · primeira vez
- [SCHEDULE.md](SCHEDULE.md) · slot booking
- [CONTRIBUTING.md](CONTRIBUTING.md) · PR style
- [docs/plan/iterate_labs.html](docs/plan/iterate_labs.html) · plano MVP completo
- [docs/architecture.md](docs/architecture.md) · arquitetura modelo
- [docs/glossary.md](docs/glossary.md) · termos por extenso
