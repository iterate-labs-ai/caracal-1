# Caracal-1

Modelo especialista em ciberseguranca de 3 bilhoes de parametros. Base Qwen2.5-Coder-3B-Instruct.

## Primeira vez aqui?

**Setup 15 minutos:** [SETUP.md](SETUP.md) · so Kaggle, sem HF, sem W&B, sem Secrets.

## Sprint 0 ativo · Caracal Base 3B v0

5 sessoes de 5h, uma por fundador. Cada um faz a sua. Pesos vivem em Kaggle Datasets publicos.

| Sessao | Fundador | Quando BRT | Steps | Issue |
|---|---|---|---|---|
| 1 | Pedro (PAMF2) | seg 24 jun 14h-19h | 0 → 4000 | [T7](https://github.com/iterate-labs-ai/caracal-1/issues/21) |
| 2 | Arthur (arturpn1) | seg 24 jun 20h-01h | 4000 → 8000 | [T8](https://github.com/iterate-labs-ai/caracal-1/issues/19) |
| 3 | Vitor (VitorScrt) | ter 25 jun 09h-14h | 8000 → 12000 | [T9](https://github.com/iterate-labs-ai/caracal-1/issues/26) |
| 4 | Kevin (dev-knz) | ter 25 jun 15h-20h | 12000 → 16000 | T11 (nova) |
| 5 | Alexandre (aletlucas) | ter 25 jun 21h-02h | 16000 → 20000 (FINAL) | T12 (nova) |
| Avaliacao | qualquer livre | quarta 26 jun | — | [T10](https://github.com/iterate-labs-ai/caracal-1/issues/17) |

Datasets de treino: PrimeVul + BigVul + DiverseVul (HuggingFace publicos, ~500M tokens).

Schedule completo: [SCHEDULE.md](SCHEDULE.md) · Issues: [pool](https://github.com/iterate-labs-ai/caracal-1/issues)

## Milestones

- [v0 · Caracal Base 3B (Sprint 0)](https://github.com/iterate-labs-ai/caracal-1/milestone/1) · ate 8 jul
- [v1 · 5 modulos LoRA + protocolo](https://github.com/iterate-labs-ai/caracal-1/milestone/2) · ate 22 jul
- [v2 · Sandbox + RL](https://github.com/iterate-labs-ai/caracal-1/milestone/3) · ate 12 ago
- [v3 · Recursive harness + ship](https://github.com/iterate-labs-ai/caracal-1/milestone/4) · ate 31 ago

## Pra cada fundador

| Acao | Onde |
|---|---|
| Setup | [SETUP.md](SETUP.md) (15 min · so Kaggle) |
| Workflow diario | [HOWTO.md](HOWTO.md) |
| Bookar slot Kaggle | [SCHEDULE.md](SCHEDULE.md) |
| Pegar tarefa | [issues](https://github.com/iterate-labs-ai/caracal-1/issues) |
| Plano completo | [docs/plan/iterate_labs.html](docs/plan/iterate_labs.html) · [PDF](docs/plan/iterate_labs.pdf) |

## Branches

| Branch | Pra que |
|---|---|
| `main` | Release estavel (futuro) |
| `dev` | Default · onde tudo mergeia |
| `t1-work` ... `t10-work` | Pre-criadas por tarefa Sprint 0 |
| `book-slot-N` | PR pequena pra bookar slot |
| `tN-curta-descricao` | Feature branches livres |

## Como rodar treino (na sua sessao)

1. Abrir https://www.kaggle.com/code · novo notebook
2. Copiar codigo de [train/notebooks/kaggle_continued_pretrain.ipynb](train/notebooks/kaggle_continued_pretrain.ipynb)
3. Editar 5 variaveis topo (SESSION, RESUME_DATASET, OUTPUT_DATASET_SLUG, STEPS, FOUNDER_HANDLE)
4. Run all
5. ~5h depois publica Kaggle Dataset publico
6. PR atualizando SCHEDULE.md trocando pending -> done

## Roteiro v0 -> v1 -> v2 -> v3

- **v0** (Sprint 0, 2 semanas) · Caracal Base 3B SFT continued pretrain · este sprint
- **v1** · 5 modulos LoRA + protocolo entre eles
- **v2** · sandbox Docker + RL com recompensa verificavel
- **v3** · recursive harness + zero-day hunt + paper arXiv

## Estrutura

```
caracal-1/
├── README.md SETUP.md HOWTO.md SCHEDULE.md CONTRIBUTING.md
├── docs/
│   ├── plan/iterate_labs.html  · plano MVP completo
│   ├── architecture.md
│   ├── recursive_harness.md   (pra v3)
│   ├── learning_trail.md
│   └── glossary.md
├── modules/    5 LoRA specialists stubs (pra v1)
├── harness/    kernel-D, sandbox, state_machine, stop_pattern (pra v2)
├── archive/    Darwin Godel + MAP-Elites (pra v2)
├── train/      continued_pretrain, sft_module, rl_grpo + configs + notebooks
├── eval/       run_probe, run_cybergym, check_decontamination
├── data/       corpus_manifest
├── infra/      compute_inventory
├── scripts/    preflight, sigterm_handler, plot_loss
└── tests/
```

Um repo so. Tudo aqui.

## Org GitHub

[iterate-labs-ai](https://github.com/iterate-labs-ai) · pool aberto

## Licenca

Apache 2.0 (heranca Qwen base)
