# Caracal-1

Modelo especialista em ciberseguranca de 3 bilhoes de parametros. Base Qwen2.5-Coder-3B-Instruct (Apache 2.0). 5 modulos LoRA especialistas sobre um Caracal Base compartilhado.

## Primeira vez aqui?

**Comece em [SETUP.md](SETUP.md)** · setup em 20 minutos.

## Status

- v1 ship alvo: 21 jul 2026
- Sprint atual: 1 (foundation)
- Tasks pool: [issues abertas](https://github.com/iterate-labs-ai/caracal-1/issues)
- Schedule placa de video: [SCHEDULE.md](SCHEDULE.md)
- Workflow diario: [HOWTO.md](HOWTO.md)

## Modulos

1. **Recon** · scout codebase + embodied gdb live
2. **Hypothesizer** · 3 hipoteses de bug class por target
3. **Crafter** · escreve PoC via state machine policy
4. **Validator** · spec formal + Z3 bounded
5. **Patcher** · unified diff + anti-supressao

## Arquitetura recursiva

6 loops aninhados (A pretrain · B SFT · C RL · D scaffold mutation · E reward coef · F humanos promovem kernel). KERNEL-D intocavel pelo loop. Ver [docs/recursive_harness.md](docs/recursive_harness.md).

## Repos relacionados (mesma org)

- [evaluator](https://github.com/iterate-labs-ai/evaluator) · suite held-out kernel D
- [archive](https://github.com/iterate-labs-ai/archive) · DGM + MAP-Elites
- [harness](https://github.com/iterate-labs-ai/harness) · runtime + sandbox + classes
- [infra](https://github.com/iterate-labs-ai/infra) · spot + creditos
- [research-scraper](https://github.com/iterate-labs-ai/research-scraper) · HF + arXiv daily

## Estrutura

```
caracal-1/
├── README.md SETUP.md HOWTO.md SCHEDULE.md CONTRIBUTING.md
├── docs/        architecture, recursive_harness, learning_trail, glossary
├── modules/     5 LoRA specialists (recon, hypothesizer, crafter, validator, patcher)
├── harness/     kernel-D, sandbox, state_machine, stop_pattern_detector
├── archive/     Darwin Godel + MAP-Elites + primitive library
├── train/       continued_pretrain, sft_module, rl_grpo + configs + notebooks
├── eval/        run_probe, run_cybergym, check_decontamination, held_out_2026
├── data/        corpus_manifest (20 fontes), decontamination
├── infra/       compute_inventory, decisions, baselines, credits
├── scripts/     preflight, sigterm_handler, push_checkpoint, plot_loss
└── tests/
```

## Licenca

Apache 2.0 (heranca Qwen base)
