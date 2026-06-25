# Caracal-1

Modelo especialista em ciberseguranca de 3 bilhoes de parametros. Base Qwen2.5-Coder-3B-Instruct.

## Primeira vez aqui?

**Setup 30 minutos:** [SETUP.md](SETUP.md)

## Sprint 0 ativo · Caracal Base 3B v0

Treinando agora. Plano enxuto, gratis (so Kaggle T4 + HuggingFace + W&B).

10 issues abertas no [milestone v0](https://github.com/iterate-labs-ai/caracal-1/milestone/1) (T1-T10).

| Sessao | Quem | Quando |
|---|---|---|
| [T7](https://github.com/iterate-labs-ai/caracal-1/issues/21) Sessao 1 (0 -> 7000) | Pedro (PAMF2) | seg 24 jun 14h-23h BRT |
| [T8](https://github.com/iterate-labs-ai/caracal-1/issues/19) Sessao 2 (7000 -> 14000) | Arthur (arturpn1) | ter 25 jun 09h-18h |
| [T9](https://github.com/iterate-labs-ai/caracal-1/issues/26) Sessao 3 (14000 -> final) | Vitor (VitorScrt) | ter 25 jun 19h-04h |
| [T10](https://github.com/iterate-labs-ai/caracal-1/issues/17) Avaliacao + debug | Kevin + Alexandre | rolling |

Datasets: PrimeVul + BigVul + DiverseVul (HuggingFace publicos, ~500M tokens).

Schedule completo: [SCHEDULE.md](SCHEDULE.md) · Issues: [pool](https://github.com/iterate-labs-ai/caracal-1/issues)

## Milestones

- [v0 · Caracal Base 3B (Sprint 0)](https://github.com/iterate-labs-ai/caracal-1/milestone/1) · ate 8 jul · 10 issues
- [v1 · 5 modulos LoRA + protocolo](https://github.com/iterate-labs-ai/caracal-1/milestone/2) · ate 22 jul
- [v2 · Sandbox + RL](https://github.com/iterate-labs-ai/caracal-1/milestone/3) · ate 12 ago
- [v3 · Recursive harness + ship](https://github.com/iterate-labs-ai/caracal-1/milestone/4) · ate 31 ago

## Pra cada fundador

| Acao | Onde |
|---|---|
| Setup | [SETUP.md](SETUP.md) (30 min) |
| Workflow diario | [HOWTO.md](HOWTO.md) |
| Bookar slot Kaggle | [SCHEDULE.md](SCHEDULE.md) |
| Pegar tarefa | [issues](https://github.com/iterate-labs-ai/caracal-1/issues) |
| Plano completo | [docs/plan/iterate_labs.html](docs/plan/iterate_labs.html) · [PDF](docs/plan/iterate_labs.pdf) |

## Branches

| Branch | Pra que |
|---|---|
| `main` | Release estavel (futuro) |
| `dev` | Default · onde tudo mergeia |
| `t1-work` ... `t10-work` | Pre-criadas por tarefa Sprint 0 (10 branches) |
| `book-slot-N` | PR pequena pra bookar slot |
| `tN-curta-descricao` | Feature branches livres |

## Como rodar treino (na sua sessao)

1. Abrir [train/notebooks/kaggle_continued_pretrain.ipynb](train/notebooks/kaggle_continued_pretrain.ipynb) no Kaggle
2. Editar 5 variaveis no topo (SESSION, RESUME, OUTPUT, STEPS, HANDLE)
3. Configurar Kaggle Secrets: HF_TOKEN + WANDB_API_KEY
4. Run all
5. ~9h depois push automatico HF Hub
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
├── train/      continued_pretrain (funcional), sft_module, rl_grpo + configs + notebooks
├── eval/       run_probe, run_cybergym, check_decontamination (funcional)
├── data/       corpus_manifest
├── infra/      compute_inventory
├── scripts/    preflight, sigterm_handler, push_checkpoint, plot_loss
└── tests/
```

Um repo so. Tudo aqui.

## Org GitHub

[iterate-labs-ai](https://github.com/iterate-labs-ai) · pool aberto

## Licenca

Apache 2.0 (heranca Qwen base)
