# SCHEDULE · revezamento de placa de video

## Inventario de contas

### Kaggle · 5 contas · 100h/sem T4 (20h cada)

| Conta Kaggle | Fundador | Phone verified | GPU T4 ativada |
|---|---|---|---|
| 1 | Pedro (PAMF2) | pendente | pendente |
| 2 | Arthur (arturpn1) | pendente | pendente |
| 3 | Vitor (VitorScrt) | pendente | pendente |
| 4 | Kevin (dev-knz) | pendente | pendente |
| 5 | Alexandre (aletlucas) | pendente | pendente |

### Colab · 5 contas Google free · ~105h/sem T4 esporadico

| Conta Google Colab | Fundador | Status |
|---|---|---|
| 1 | Pedro | pendente |
| 2 | Arthur | pendente |
| 3 | Vitor | pendente |
| 4 | Kevin | pendente |
| 5 | Alexandre | pendente |

Pedro tem adicionalmente **4 contas Colab extras** = pode rodar ate 5 sessoes Colab paralelas no Pedro sozinho. Overflow capacity.

### Capacidade total v0

- Kaggle T4: 100h/sem garantido
- Colab T4: ~105h/sem esporadico + 84h/sem extras do Pedro
- Total: ~290h/sem T4

## Sprint 0 · Caracal Base 3B v0 · 25h em 3 sessoes Kaggle

Treino MVP: Qwen2.5-Coder-3B + PrimeVul + BigVul + DiverseVul. ~20000 passos.

| Sessao | Conta | Inicio BRT | Fim BRT | Resume | Output | Status |
|---|---|---|---|---|---|---|
| 1 | Pedro (PAMF2) | seg 24 jun 14h00 | seg 24 jun 23h00 | (do zero) | step-7000 | pending |
| 2 | Arthur (arturpn1) | ter 25 jun 09h00 | ter 25 jun 18h00 | step-7000 | step-14000 | pending |
| 3 | Vitor (VitorScrt) | ter 25 jun 19h00 | qua 26 jun 04h00 | step-14000 | **caracal-base-3b-v0 (final)** | pending |

**Kevin e Alexandre observam, debugam se quebrar, ajudam revisao.**

## Sprint 0.5 · Avaliacao basica

Apos sessao 3 terminar, qualquer fundador disponivel:

| Tarefa | Conta | Tempo estimado | Quem pega |
|---|---|---|---|
| Avaliar Caracal Base 3B v0 vs Qwen base zero-shot em 50 vulns CyberGym | Kaggle (qualquer) | 4h T4 | livre |
| Documentar resultados em `infra/baselines/v0_eval.md` | local | 1h | livre |
| Decidir: continuar caminho ou repensar | reuniao | 30min | todos |

## Apos v0 funcionar · decisao do time

Possibilidades:

| Cenario | Proximo passo |
|---|---|
| Score sobe vs Qwen base | Adicionar mais datasets (scraping NVD/Project Zero) · v1 |
| Score nao move | Repensar hyperparams ou abordagem |
| Score cai | Debug contaminacao ou hyperparam |
| Treino quebra | Bug no script · debug + retry |

## Como bookar slot futuro

1. Editar este arquivo na branch `dev`
2. Trocar `pending` por `in-progress · seu_handle · timestamp`
3. PR pequena
4. Self-merge se nao conflito
5. Apos terminar: trocar para `done · revision step-XXXX`

## HF Hub revisions

Todos checkpoints vao como revisions em `iterate-labs/caracal-base-3b-v0`:

- step-7000 (pos sessao 1)
- step-14000 (pos sessao 2)
- main / final (pos sessao 3)

Audit: `huggingface-cli list-revisions iterate-labs/caracal-base-3b-v0`
