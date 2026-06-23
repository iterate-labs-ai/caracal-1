# HOWTO · como usar o repo Caracal-1

## Setup inicial (uma vez)

```bash
# 1. Clone
git clone https://github.com/iterate-labs-ai/caracal-1
cd caracal-1
git checkout dev   # branch default de trabalho

# 2. Install
pip install -r requirements.txt

# 3. Login HuggingFace (token de escrita em huggingface.co/settings/tokens)
huggingface-cli login

# 4. Login Weights and Biases
wandb login

# 5. Login Kaggle
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 6. Login Modal (opcional, so quem mexe em sandbox)
modal token new

# 7. Verifica tudo OK
bash scripts/preflight.sh
```

## Workflow diario

1. Olhar [pool de tarefas](https://github.com/iterate-labs-ai/caracal-1/issues) abertas
2. Filtrar por `sprint-1` + sem assignee + prioridade `MUST` antes de `STRETCH`
3. Comentar `claim` na issue para auto-assignar
4. Olhar [SCHEDULE.md](SCHEDULE.md) para slots de placa de video livres
5. Bookar slot proximo via PR pequena editando SCHEDULE.md
6. Abrir notebook Kaggle ou Colab autenticando HF + W&B
7. Rodar pre-flight `bash scripts/preflight.sh`
8. Iniciar tarefa

## Workflow durante treino (cada sessao Kaggle ou Colab)

1. Pull checkpoint mais recente

```bash
huggingface-cli download iterate-labs/MODELO \
    --revision step-XXXX --local-dir ./ckpt
```

2. Iniciar W&B run resumindo o anterior

```python
import wandb
wandb.init(project="caracal-base-pretrain", resume="allow",
           id="caracal-base-pretrain-main")
```

3. Rodar script de treino com flag `--resume-from ./ckpt`
4. SIGTERM handler salva final automatico se Kaggle ou Colab morre
5. Push checkpoint a cada 200 passos
6. Ao terminar 9h ou steps planejados: push final, atualiza SCHEDULE.md, sinaliza proximo no Discord

```bash
huggingface-cli upload iterate-labs/MODELO \
    ./ckpt-out . --revision step-YYYY \
    --commit-message "step YYYY by $(whoami) at $(date)"
```

## Workflow PR

1. Branch local: `git checkout -b cN-meu-fix dev`
2. Commit mensagens claras
3. Push: `git push -u origin cN-meu-fix`
4. Abrir PR contra `dev` linkando issue: `Closes #N`
5. Pedir review de outro fundador que NAO escreveu o codigo
6. Issues com tag `kernel-D`: 2 reviewers obrigatorios + ADR
7. Apos merge: `done` na issue, claim proxima

## Branches

| Branch | Pra que | Quem pode push direto |
|---|---|---|
| `main` | release estavel · so PRs de dev aprovadas | ninguem direto (PR-only) |
| `dev` | branch default de trabalho · onde features mergeam primeiro | qualquer fundador |
| `exp/<nome>` | branches experimentais por fundador | dono da branch |
| `release/v1` | snapshot pre-ship congelado | so Pedro libera |

## Sandbox local

```bash
modal serve sandbox/caracal_sandbox.py

modal run sandbox/caracal_sandbox.py::run_poc \
    --vuln-id cybergym-poc-42 \
    --poc-payload "..."

modal run sandbox/test_attacks.py
```

## Avaliacao

```bash
# Probe set rapido (100 prompts)
python eval/run_probe.py \
    --adapter iterate-labs/caracal-1-crafter-lora \
    --base iterate-labs/caracal-base-3b \
    --out eval/reports/probe-step-XXXX.json

# CyberGym slice 50
python eval/run_cybergym.py --subset slice-50 --adapter ...

# Full CyberGym 1507 (so dia 20 e dia 28)
python eval/run_cybergym.py --subset full-1507 --adapter ... \
    --temp 0.8 --top-p 0.95 --k 5
```

## Cheat sheet rapido

```bash
# Status treino
huggingface-cli list-files iterate-labs/caracal-base-pretrain | tail -5
wandb runs --project iterate-labs/caracal-base-pretrain | head

# Quem ta no slot agora
grep "in-progress" SCHEDULE.md

# Loss curve
python scripts/plot_loss.py --run-id caracal-base-pretrain --last 24h

# STOP-pattern flags
ls harness/incidents/

# Custo Modal
modal billing status
```

## Quem chamar quando

| Situacao | Quem |
|---|---|
| Treino travou | Discord #caracal-relay + screenshot |
| HF Hub push falhou | Discord #caracal-relay + log |
| Slot conflict | Primeiro book wins, perdedor re-book proximo |
| Bug kernel-D | Issue urgente · 2 fundadores debug juntos |
| STOP-pattern flag | Para todo treino. 2 humanos review. |
| Sandbox attack test falhou | BLOCKER. Para tudo. 2 reviewers + ADR. |
| Modal custo >$200/dia | Auto alerta. Pausa sandbox. Review config. |

## Documentos importantes

- `README.md` · visao geral
- `HOWTO.md` · este arquivo
- `SCHEDULE.md` · slot booking placa de video
- `CONTRIBUTING.md` · contribuicao + PR style
- `docs/architecture.md` · arquitetura modelo
- `docs/recursive_harness.md` · 6 loops
- `docs/learning_trail.md` · links de aprendizado
- `docs/glossary.md` · termos por extenso
- `infra/decisions/` · ADRs
- `eval/held_out_2026.yaml` · KERNEL-D, NUNCA tocar
- `harness/kernel/` · KERNEL-D, 2 humanos editam
- `harness/incidents/` · STOP-pattern flags
