# Caracal-1

Modelo especialista em ciberseguranca de 3 bilhoes de parametros. Base Qwen2.5-Coder-3B-Instruct.

## Primeira vez aqui?

**Setup 15 minutos:** [SETUP.md](SETUP.md)

## Sprint 0 ativo · Caracal Base 3B v0

Treinando agora. Plano enxuto, gratis (so Kaggle T4 + HuggingFace + W&B).

| O que | Quem | Quando |
|---|---|---|
| Sessao 1 (0 → 7000 passos) | Pedro (PAMF2) | seg 24 jun 14h-23h BRT |
| Sessao 2 (7000 → 14000) | Arthur (arturpn1) | ter 25 jun 09h-18h |
| Sessao 3 (14000 → final) | Vitor (VitorScrt) | ter 25 jun 19h-04h |
| Avaliacao + debug | Kevin + Alexandre | rolling |

Datasets: PrimeVul + BigVul + DiverseVul (HuggingFace publicos, ~500M tokens). Sem scraping.

Schedule completo: [SCHEDULE.md](SCHEDULE.md)

## Pra cada fundador

| Acao | Onde |
|---|---|
| Setup | [SETUP.md](SETUP.md) (15 min) |
| Workflow diario | [HOWTO.md](HOWTO.md) |
| Bookar slot Kaggle | [SCHEDULE.md](SCHEDULE.md) |
| Pegar tarefa | [issues](https://github.com/iterate-labs-ai/caracal-1/issues) |
| Plano completo MVP | [docs/plan/iterate_labs.html](docs/plan/iterate_labs.html) |

## Como rodar treino (na sua sessao)

1. Abrir [train/notebooks/kaggle_continued_pretrain.ipynb](train/notebooks/kaggle_continued_pretrain.ipynb) no Kaggle
2. Editar 5 variaveis no topo (SESSION_NUMBER, RESUME_REVISION, OUTPUT_REVISION, STEPS_TO_RUN, FOUNDER_HANDLE)
3. Configurar Kaggle Secrets: HF_TOKEN + WANDB_API_KEY
4. Run all
5. ~9h depois push automatico HF Hub
6. PR atualizando SCHEDULE.md trocando pending → done

## Apos v0 funcionar

Time decide proximos passos baseado em avaliacao. Veja docs/plan/iterate_labs.html secao 8.

## Estrutura

```
caracal-1/
├── README.md SETUP.md HOWTO.md SCHEDULE.md CONTRIBUTING.md
├── docs/
│   ├── plan/iterate_labs.html  ← plano MVP completo
│   ├── architecture.md
│   ├── recursive_harness.md   (pra v2)
│   ├── learning_trail.md
│   └── glossary.md
├── modules/    5 LoRA specialists stubs (pra Sprint 2)
├── harness/    kernel-D, sandbox, state_machine, stop_pattern
├── archive/    Darwin Godel + MAP-Elites (pra Sprint 3)
├── train/      continued_pretrain (funcional), sft_module, rl_grpo + configs + notebooks
├── eval/       run_probe, run_cybergym, check_decontamination (funcional)
├── data/       corpus_manifest
├── infra/      compute_inventory
├── scripts/    preflight, sigterm_handler, push_checkpoint, plot_loss
└── tests/
```

## Org GitHub

[iterate-labs-ai](https://github.com/iterate-labs-ai) · 9 repos · pool aberto

## Licenca

Apache 2.0 (heranca Qwen base)
