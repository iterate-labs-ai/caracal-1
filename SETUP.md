# SETUP · primeira vez no Caracal-1

Tempo total: ~15 minutos. So Kaggle. Sem HuggingFace, sem Weights and Biases, sem Secrets.

## Pre-requisitos

- Conta GitHub aceita convite em https://github.com/iterate-labs-ai
- Email cadastrado
- Telefone disponivel pra phone verification do Kaggle

## Passo 1 · Aceitar convite GitHub (1 min)

https://github.com/orgs/iterate-labs-ai/invitations

Se nao chegou, confirma com o Pedro qual handle GitHub voce usa.

## Passo 2 · Conta Kaggle phone-verified + GPU T4 (10 min)

1. Criar conta em https://www.kaggle.com/account/login (ou login Google)
2. Phone verification em https://www.kaggle.com/settings · Phone verify (~5 min pro SMS chegar)
3. Ativar GPU T4 em https://www.kaggle.com/settings · aceitar terms

## Passo 3 · Instalar Kaggle CLI local (2 min)

Pra rodar treino do PC e baixar checkpoint:

```bash
pip install kaggle
```

Baixar `kaggle.json` em https://www.kaggle.com/settings (Create New API Token):

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Testar
kaggle datasets list
```

## Passo 4 · Clonar repo (2 min)

```bash
git clone https://github.com/iterate-labs-ai/caracal-1
cd caracal-1
git checkout dev
```

## Checklist final

| Item | OK |
|---|---|
| Aceitei convite GitHub iterate-labs-ai | ☐ |
| Conta Kaggle phone-verified | ☐ |
| GPU T4 ativada em Kaggle settings | ☐ |
| Kaggle CLI instalado + `kaggle.json` em ~/.kaggle/ | ☐ |
| Clonei repo caracal-1 | ☐ |

## Troubleshooting

| Problema | Solucao |
|---|---|
| Phone verify Kaggle nao chega | Esperar 10 min · tentar outro numero · usar Google login |
| Kaggle nao da GPU T4 | Esperar algumas horas · pode ser quota |
| `kaggle datasets list` da erro auth | Conferir `kaggle.json` em ~/.kaggle/ com permissao 600 |

## Pra rodar treino na sua sessao

Abrir https://www.kaggle.com/code · novo notebook · copiar conteudo de [train/notebooks/kaggle_continued_pretrain.ipynb](train/notebooks/kaggle_continued_pretrain.ipynb). Detalhes em [HOWTO.md](HOWTO.md).
