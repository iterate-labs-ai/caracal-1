# Caracal-1

Modelo especialista em ciberseguranca de 3 bilhoes de parametros. Base Qwen2.5-Coder-3B-Instruct.

## Primeira vez aqui?

**Setup 30 minutos:** [SETUP.md](SETUP.md)

## Sprint 0 ativo · Caracal Base 3B v0

Treinando agora. Plano enxuto, gratis (so Kaggle T4 + HuggingFace + W&B).

| Sessao | Quem | Quando |
|---|---|---|
| [T7](https://github.com/iterate-labs-ai/caracal-1/issues/23) Sessao 1 (0 → 7000) | Pedro (PAMF2) | seg 24 jun 14h-23h BRT |
| [T8](https://github.com/iterate-labs-ai/caracal-1/issues/24) Sessao 2 (7000 → 14000) | Arthur (arturpn1) | ter 25 jun 09h-18h |
| [T9](https://github.com/iterate-labs-ai/caracal-1/issues/25) Sessao 3 (14000 → final) | Vitor (VitorScrt) | ter 25 jun 19h-04h |
| [T10](https://github.com/iterate-labs-ai/caracal-1/issues/17) Avaliacao + debug | Kevin + Alexandre | rolling |

Datasets: PrimeVul + BigVul + DiverseVul (HuggingFace publicos, ~500M tokens).

Schedule completo: [SCHEDULE.md](SCHEDULE.md) · Issues: [pool](https://github.com/iterate-labs-ai/caracal-1/issues)

## Repos relacionados

- **caracal-1** (este) · treino · pesos vivem no HF Hub
- [caracal-cyber](https://github.com/iterate-labs-ai/caracal-cyber) · sandbox Docker + CyberGym eval + zero-day hunt + red-team
- [evaluator](https://github.com/iterate-labs-ai/evaluator) · suite held-out (legacy)
- [archive](https://github.com/iterate-labs-ai/archive) · DGM + MAP-Elites (futuro v2)
- [harness](https://github.com/iterate-labs-ai/harness) · runtime + sandbox (futuro v2)
- [infra](https://github.com/iterate-labs-ai/infra) · spot + creditos
- [research-scraper](https://github.com/iterate-labs-ai/research-scraper) · HF + arXiv daily cron

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
6. PR atualizando SCHEDULE.md trocando pending → done

## Roteiro v0 → v1 → v2 → v3

- **v0** (Sprint 0, 2 semanas) · Caracal Base 3B SFT continued pretrain · este sprint
- **v1** · 5 modulos LoRA + protocolo entre eles
- **v2** · sandbox Docker em caracal-cyber + RL com recompensa verificavel
- **v3** · recursive harness + zero-day hunt + paper arXiv

## Org GitHub

[iterate-labs-ai](https://github.com/iterate-labs-ai) · 10 repos · pool aberto

## Licenca

Apache 2.0 (heranca Qwen base)
