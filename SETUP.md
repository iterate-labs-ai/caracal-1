# SETUP · primeira vez no Caracal-1

Tempo total: ~20 minutos. Faz uma vez no inicio.

## Pre-requisitos

- Conta GitHub aceita convite em https://github.com/iterate-labs-ai
- Python 3.11 ou 3.12 instalado
- Git instalado
- ~10 GB livres em disco
- Email cadastrado

## Passo 1 · Clonar o repo

```bash
git clone https://github.com/iterate-labs-ai/caracal-1
cd caracal-1
git checkout dev   # dev e o branch default de trabalho
```

## Passo 2 · Instalar dependencias

```bash
# Criar ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .\.venv\Scripts\activate    # Windows

# Atualizar pip
pip install --upgrade pip

# Instalar pacotes
pip install -r requirements.txt
```

## Passo 3 · Criar contas necessarias

### 3.1 HuggingFace (obrigatorio)

1. Criar conta em https://huggingface.co/join
2. Aceitar convite para org `iterate-labs` (Pedro envia)
3. Gerar token de escrita em https://huggingface.co/settings/tokens
   - Tipo: **Write**
   - Nome: `caracal-1-dev`
4. Login local:

```bash
huggingface-cli login
# cole o token quando pedir
```

### 3.2 Weights and Biases (obrigatorio)

1. Criar conta em https://wandb.ai/signup
2. Aceitar convite para org `iterate-labs` (Pedro envia)
3. Login local:

```bash
wandb login
# cole o token de https://wandb.ai/authorize
```

### 3.3 Kaggle (obrigatorio · 1 conta por fundador)

1. Criar conta em https://www.kaggle.com/account/login (se ainda nao tem)
2. Phone verification em https://www.kaggle.com/settings (necessario pra placa de video)
3. Ativar T4 GPU em https://www.kaggle.com/settings (aceitar terms)
4. Baixar `kaggle.json` em https://www.kaggle.com/settings (Create New API Token)
5. Mover pra local certo:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3.4 Colab Pro+ (recomendado · 4 contas por fundador)

1. Criar 4 contas Google (use aliases tipo `pedrocaracal1@gmail.com`, `pedrocaracal2@gmail.com`)
2. Assinar Colab Pro+ em cada conta · https://colab.research.google.com/signup
   - Custo: $50/mes por conta
   - Total: $1000/mes (5 pessoas × 4 contas × $50)
3. Salvar credenciais em password manager pessoal

### 3.5 Modal (opcional · so quem mexe em sandbox)

1. Criar conta em https://modal.com/signup
2. Aceitar convite para workspace `iterate-labs` (Pedro envia)
3. Login local:

```bash
modal token new
```

## Passo 4 · Validar setup

```bash
bash scripts/preflight.sh
```

Deve mostrar:

```
==> preflight check
checking huggingface-cli auth... OK
checking wandb auth... OK
checking python deps... OK · torch 2.5.x, CUDA: True/False
checking GPU... <info da placa ou WARN CPU>
checking disk space... <livre>
checking KERNEL-D integrity... OK
==> preflight OK · ready to train
```

Se algum item falhar, ver mensagem de erro e refazer o passo correspondente.

## Passo 5 · Confirmar acesso ao repo

```bash
gh auth status   # se nao tem gh CLI: brew install gh OU apt install gh
```

Deve mostrar logado na conta GitHub que aceitou convite. Senao:

```bash
gh auth login
```

## Passo 6 · Ler documentos chave

Antes de pegar tarefa, ler:

1. [README.md](README.md) · visao geral 2 min
2. [HOWTO.md](HOWTO.md) · workflow diario 5 min
3. [SCHEDULE.md](SCHEDULE.md) · slot booking de placa de video 2 min
4. [CONTRIBUTING.md](CONTRIBUTING.md) · PR style + KERNEL-D rules 5 min
5. [docs/architecture.md](docs/architecture.md) · arquitetura modelo 5 min
6. [docs/recursive_harness.md](docs/recursive_harness.md) · 6 loops self-improving 5 min
7. [docs/learning_trail.md](docs/learning_trail.md) · links de aprendizado (consultar conforme precisar)
8. [docs/glossary.md](docs/glossary.md) · termos por extenso (consultar)

## Passo 7 · Pegar primeira tarefa

1. Abrir https://github.com/iterate-labs-ai/caracal-1/issues
2. Filtrar por label `sprint-1` + sem assignee + tag `MUST`
3. Ler issue, decidir se voce consegue pegar
4. Comentar `claim` na issue para auto-assignar
5. Abrir branch local:

```bash
git checkout -b cN-curta-descricao dev
```

6. Trabalhar
7. PR contra `dev` quando pronto: `Closes #N`

## Passo 8 · Bookar slot de placa de video (Kaggle/Colab)

1. Abrir [SCHEDULE.md](SCHEDULE.md)
2. Encontrar slot livre (status `pending`)
3. Editar em branch local: trocar `pending` para `in-progress · seu_handle · timestamp`
4. PR pequena: `book slot N`
5. Self-merge se nao conflito
6. Sinalizar Discord `#caracal-relay`: "estou no slot N agora"

## Passo 9 · Rodar primeiro treino (smoke test)

Notebook template Kaggle em `train/notebooks/kaggle_continued_pretrain.ipynb`. Abrir Kaggle, criar novo notebook, importar este e rodar primeira celula:

```python
# Cell 1 · auth check
!huggingface-cli whoami
!wandb status
print("smoke OK · pronto pra rodar treino completo")
```

Se passou, esta tudo certo. Pronto para Sprint 1.

## Troubleshooting

### CUDA out of memory

Reduzir `per_device_train_batch_size` para 2 em `train/configs/caracal_base_3b.yaml`.

### HF Hub push falhou

Token expirou ou nao tem permissao write. Refazer passo 3.1.

### Kaggle sessao morreu antes do final

SIGTERM handler ja foi instalado. Checkpoint final salva automatico em HF Hub. Proximo no relay resume de la.

### Modal cobrou caro

Editar `harness/kernel/budgets.yaml` (KERNEL-D, exige PR + 2 reviewers). Default $200/dia alerta.

### Conflito de slot no SCHEDULE

Primeiro a mergeear ganha. Quem perdeu pega proximo livre.

## Quem chamar quando

| Situacao | Quem |
|---|---|
| Setup falhou em algum passo | Discord #caracal-help + screenshot |
| Conta convite nao chegou (HF/W&B/Modal/GitHub) | Pedro (PAMF2) |
| Pago Colab Pro+ e ainda nao tem A100 | Esperar 24h, Google libera gradual |
| Bug no codigo | Issue urgente · 2 fundadores debug juntos |

## Proxima etapa

Apos setup completo, leia [HOWTO.md](HOWTO.md) workflow diario. Bora trabalhar.
