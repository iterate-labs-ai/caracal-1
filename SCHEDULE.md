# SCHEDULE · revezamento de placa de video

## Inventario de contas Kaggle

5 contas Kaggle, 1 por fundador, ~20h/sem T4 cada = 100h/sem garantido.

| Conta Kaggle | Fundador | Phone verified | T4 ativada |
|---|---|---|---|
| 1 | Pedro (PAMF2) | pendente | pendente |
| 2 | Arthur (arturpn1) | pendente | pendente |
| 3 | Vitor (VitorScrt) | pendente | pendente |
| 4 | Kevin (dev-knz) | pendente | pendente |
| 5 | Alexandre (aletlucas) | pendente | pendente |

## Sprint 0 · Caracal Base 3B v0 · 5 sessoes de 5h cada (25h total)

Cada fundador faz 1 sessao. Sem ninguem observando. Cada um salva checkpoint como Kaggle Dataset.

| Sessao | Conta | Quando BRT | Steps | Resume Kaggle Dataset | Output Kaggle Dataset |
|---|---|---|---|---|---|
| 1 | Pedro (PAMF2) | seg 24 jun 14h-19h | 0 -> 4000 | (do zero) | `pamf2/caracal-base-step4000` |
| 2 | Arthur (arturpn1) | seg 24 jun 20h-01h | 4000 -> 8000 | `pamf2/caracal-base-step4000` | `arturpn1/caracal-base-step8000` |
| 3 | Vitor (VitorScrt) | ter 25 jun 09h-14h | 8000 -> 12000 | `arturpn1/caracal-base-step8000` | `vitorscrt/caracal-base-step12000` |
| 4 | Kevin (dev-knz) | ter 25 jun 15h-20h | 12000 -> 16000 | `vitorscrt/caracal-base-step12000` | `dev-knz/caracal-base-step16000` |
| 5 | Alexandre (aletlucas) | ter 25 jun 21h-02h | 16000 -> 20000 | `dev-knz/caracal-base-step16000` | `aletlucas/caracal-base-3b-v0` (FINAL) |

5 fundadores, 5 sessoes, 25h total. Cada um pega 5h da quota 20h/sem (sobra muito).

**Apos final:** Alexandre publica dataset `aletlucas/caracal-base-3b-v0` como nosso modelo v0. Qualquer fundador baixa pra avaliar.

## Sprint 0.5 · Avaliacao (T10)

Qualquer fundador, ~6h Kaggle T4:

| Tarefa | Conta | Tempo | Quem pega |
|---|---|---|---|
| Avaliar Caracal Base 3B v0 vs Qwen base zero-shot em 50 vulns CyberGym | Kaggle (qualquer) | ~4h T4 | livre |
| Rodar HumanEval+ regression em ambos | Kaggle | ~2h T4 | livre |
| Documentar em `infra/baselines/v0_eval.md` + abrir reuniao | local | 1h | livre |

## Apos v0 · decisao do time

Reuniao quarta 26 jun manha. Baseado no numero:

| Cenario | Proximo passo |
|---|---|
| Score sobe vs Qwen base | Continuar v1 · 5 modulos LoRA |
| Score nao move | Repensar: hyperparams, dataset, learning rate |
| Score cai | Debug contaminacao + hyperparam |
| Treino quebrou em alguma sessao | Bug no script · debug + retry |

## Como bookar slot

1. Editar este arquivo na branch `dev`
2. Trocar `pending` por `in-progress · seu_handle · timestamp`
3. PR pequena `book slot N`
4. Self-merge
5. Apos terminar: trocar por `done · dataset XXX`

## Como compartilhar checkpoint entre contas Kaggle

No final da sua sessao, o notebook automaticamente faz `kaggle datasets create` do output. Sem auth extra · cada conta cria datasets publicos da org Kaggle.

Proximo no relay roda no notebook:
```python
import kaggle
kaggle.api.dataset_download_files('username/caracal-base-stepXXXX', path='./ckpt-in', unzip=True)
```

Datasets sao publicos pra fundadores acessarem entre si sem auth.
