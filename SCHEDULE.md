# SCHEDULE · revezamento de placa de video

## Inventario de contas Kaggle

5 contas Kaggle, 1 por fundador. Cada conta tem quota de 30h GPU/sem.

| Conta Kaggle | Fundador | Phone verified | T4 x2 ativado |
|---|---|---|---|
| pedroafonso2 | Pedro | sim | sim |
| arturpn1 | Arthur | pendente | pendente |
| vitorscrt | Vitor | pendente | pendente |
| dev-knz | Kevin | pendente | pendente |
| aletlucas | Alexandre | pendente | pendente |

## Sprint 0 · Caracal Base 3B v0 · 5 sessoes de 900 steps cada

Limite Kaggle: 12h por kernel. Cada sessao ~10h em T4 x2 (~40s/step com fp16 + LoRA r=32 + seq 2048).

Cada fundador faz 1 sessao. Salva checkpoint como Kaggle Dataset publico. Proximo pull-a do anterior.

| Sessao | Conta Kaggle | Steps | Resume Dataset | Output Dataset |
|---|---|---|---|---|
| 1 | pedroafonso2 | 0 -> 900 | (cold start) | `pedroafonso2/caracal-base-3b-s01` |
| 2 | arturpn1 | 900 -> 1800 | `pedroafonso2/caracal-base-3b-s01` | `arturpn1/caracal-base-3b-s02` |
| 3 | vitorscrt | 1800 -> 2700 | `arturpn1/caracal-base-3b-s02` | `vitorscrt/caracal-base-3b-s03` |
| 4 | dev-knz | 2700 -> 3600 | `vitorscrt/caracal-base-3b-s03` | `dev-knz/caracal-base-3b-s04` |
| 5 | aletlucas | 3600 -> 4500 | `dev-knz/caracal-base-3b-s04` | `aletlucas/caracal-base-3b-v0` (FINAL) |

Total: **4500 steps** = ~95M tokens treinados em PrimeVul + BigVul + DiverseVul.

Apos final: Alexandre publica `aletlucas/caracal-base-3b-v0` como release v0. Qualquer fundador baixa pra avaliar.

## Sprint 0.5 · Avaliacao

Qualquer fundador, ~3h Kaggle T4 x2:

| Tarefa | Comando | Output |
|---|---|---|
| Probe set (51 prompts) | `python eval/run_probe.py --adapter ./ckpt-out --out probe.json` | ppl + CWE hit rate |
| CyberGym slice-50 | `python eval/run_cybergym.py --subset slice-50 --adapter ./ckpt-out` | pass@1 |
| Baseline vs Qwen | `python eval/compare_baseline.py --adapter ./ckpt-out --out baseline.json` | win_rate, v0_pass bool |

Sucesso v0 = `win_rate >= 80%` no `compare_baseline.py` (Caracal melhor ppl que Qwen em >=80% dos probes).

## Como rodar SUA sessao

1. Abrir https://www.kaggle.com/code · novo notebook
2. Copiar codigo de [train/notebooks/kaggle_continued_pretrain.ipynb](train/notebooks/kaggle_continued_pretrain.ipynb)
3. Editar 5 variaveis topo:
   ```python
   SESSION = 2                                     # numero da sua sessao
   FOUNDER_HANDLE = "arturpn1"                     # seu handle Kaggle
   RESUME_DATASET = "pedroafonso2/caracal-base-3b-s01"  # output da sessao anterior
   OUTPUT_DATASET_SLUG = "caracal-base-3b-s02"
   STEPS = 900
   ```
4. Settings: **GPU T4 x2** + Internet ON + Persistence
5. Save & Run All
6. ~10h depois dataset publicado automaticamente
7. PR atualizando este SCHEDULE.md sua linha pending -> done

## Apos v0 · decisao do time

Reuniao quando final terminar. Baseado nas metricas:

| Cenario | Proximo passo |
|---|---|
| Win rate >= 80% vs Qwen base | Continuar v1 · 5 modulos LoRA |
| Win rate 50-80% | Investigar: ajustar lr, mais steps, melhor decontam |
| Win rate < 50% | Debug serio: contaminacao, schema, eval bug |

## Como bookar slot

1. Branch `book-slot-N` a partir de `dev`
2. Editar este arquivo trocando linha vazia por `in-progress · seu_handle · timestamp`
3. PR contra `dev` · self-merge
4. Apos terminar: trocar por `done · dataset_slug · timestamp`
