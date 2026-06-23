# SCHEDULE · slot booking de placa de video

Pool de slots Kaggle (5 contas × 20h/sem T4) + Colab Pro+ (20 contas × 3h/dia). Atualizar este arquivo via PR pequena ao bookar slot. Mencionar `@PAMF2` se conflito.

## Sprint 1 · Caracal Base continued pretrain · semana 23-29 jun

Cronograma alvo: 9 sessoes × ~9h = 80h T4 distribuidas em 2-3 dias. Total ~50000 passos.

| Sessao | Conta Kaggle | Inicio BRT | Fim BRT | Steps | Checkpoint saida | Status |
|---|---|---|---|---|---|---|
| 1 | PAMF2 (Pedro) | seg 23 jun 09:00 | seg 23 jun 18:00 | 0 → 5500 | step-5500 | pending |
| 2 | arturpn1 (Arthur) | seg 23 jun 14:00 | seg 23 jun 23:00 | 5500 → 11000 | step-11000 | pending |
| 3 | VitorScrt (Vitor) | seg 23 jun 19:00 | ter 24 jun 04:00 | 11000 → 16500 | step-16500 | pending |
| 4 | dev-knz (Kevin) | ter 24 jun 00:00 | ter 24 jun 09:00 | 16500 → 22000 | step-22000 | pending |
| 5 | aletlucas (Alexandre) | ter 24 jun 05:00 | ter 24 jun 14:00 | 22000 → 27500 | step-27500 | pending |
| 6 | PAMF2 (2a) | ter 24 jun 10:00 | ter 24 jun 19:00 | 27500 → 33000 | step-33000 | pending |
| 7 | arturpn1 (2a) | ter 24 jun 15:00 | qua 25 jun 00:00 | 33000 → 38500 | step-38500 | pending |
| 8 | VitorScrt (2a) | ter 24 jun 20:00 | qua 25 jun 05:00 | 38500 → 44000 | step-44000 | pending |
| 9 | dev-knz (2a) | qua 25 jun 01:00 | qua 25 jun 10:00 | 44000 → 50000 | **step-50000 final** | pending |

Sobreposicao de 4-5h entre sessoes consecutivas garante zero gap de placa de video. Cada fundador faz 2 sessoes nesta semana = ~18h dentro do limite Kaggle 20h.

## Sprint 2 · 5 modulos SFT estagio 2 · semana 30 jun - 6 jul

Padrao diferente: cada fundador roda 1 modulo em paralelo na propria conta. Sem revezamento.

| Conta | Modulo | Dataset | Tempo estimado | Status |
|---|---|---|---|---|
| qualquer fundador disponivel | Recon | gdb_traces 5K | ~25h T4 | pending |
| qualquer fundador disponivel | Hypothesizer | ctf_writeups 8K | ~30h T4 | pending |
| qualquer fundador disponivel | Crafter | poc_patch 5K + Cyber-Zero 10K | ~40h T4 | pending |
| qualquer fundador disponivel | Validator | spec_synth gerados | ~20h T4 | pending |
| qualquer fundador disponivel | Patcher | patches_cve | ~30h T4 | pending |

Quem termina antes pega outro modulo. Pool aberto.

## Sprint 3 · RL principal · semana 7-13 jul

Move para Google Cloud A100 quando creditos chegarem ($25k via Google for Startups). Schedule mudo para slots A100. Atualizar este arquivo quando creditos confirmados.

## Sprint 4 · avaliacao + ship · semana 14-20 jul

Ultimos 100h A100 para full eval + zero-day hunt + release HF Hub.

## Como bookar slot

1. Editar este arquivo na branch `dev`
2. Trocar `pending` para `in-progress · sua conta · timestamp`
3. PR pequena: `book slot N para X`
4. Self-merge se nao conflito · sinalizar Discord #caracal-relay
5. Apos terminar treino, edit row para `done · revision step-YYYY`

## Como resolver conflito

Quem book primeiro wins (timestamp do commit). Quem perdeu re-book proximo slot livre. Documentar conflito em `infra/incidents/schedule-conflict-YYYY-MM-DD.md`.

## HF Hub revisions

Cada checkpoint vai como revision tagged em `iterate-labs/caracal-base-pretrain`. Revisions imutaveis. Audit completo via `huggingface-cli list-revisions iterate-labs/caracal-base-pretrain`.
