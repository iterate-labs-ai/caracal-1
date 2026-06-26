# Caracal-1

Modelo especialista em ciberseguranca de 3 bilhoes de parametros. Base **Qwen2.5-Coder-3B-Instruct** (Apache 2.0).

> Iterate Labs · 5 fundadores · bootstrapped · Campo Grande BR
> Org GitHub: [iterate-labs-ai](https://github.com/iterate-labs-ai)

---

## TL;DR

- **Base:** Qwen2.5-Coder-3B-Instruct (Transformer decoder-only · 36 layers · GQA 16/2 · SwiGLU · RMSNorm · RoPE base 1M · 32K context)
- **Adapter:** LoRA r=32 alpha=64 em q/k/v/o + gate/up/down
- **Treino:** Unsloth + TRL SFTTrainer · FP16 (T4 sem bf16) · packing · cosine lr=5e-5
- **Datasets:** PrimeVul + BigVul + DiverseVul (HF publicos · decontam CVE-ID antes do load)
- **Stack v0:** so Kaggle T4 x2. Sem HuggingFace Hub, sem Weights and Biases, sem Secrets. Custo $0.
- **Eval:** probe set 51 + CyberGym (Berkeley ICLR 2026, 1507 vulns) + baseline vs Qwen puro
- **Pool aberto:** todos treinam tudo. Sem dono fixo de modulo.

---

## Estado atual · Sprint 0 (v0)

5 sessoes de ~10h em Kaggle T4 x2, uma por fundador. Pesos vivem em Kaggle Datasets publicos (cada sessao publica, proxima pull-a).

| Sessao | Fundador | Steps | Issue |
|---|---|---|---|
| 1 | Pedro (pedroafonso2) | 0 -> 900 | [T7](https://github.com/iterate-labs-ai/caracal-1/issues/21) |
| 2 | Arthur (arturpn1) | 900 -> 1800 | [T8](https://github.com/iterate-labs-ai/caracal-1/issues/19) |
| 3 | Vitor (vitorscrt) | 1800 -> 2700 | [T9](https://github.com/iterate-labs-ai/caracal-1/issues/26) |
| 4 | Kevin (dev-knz) | 2700 -> 3600 | T11 |
| 5 | Alexandre (aletlucas) | 3600 -> 4500 (FINAL) | T12 |
| Avaliacao | qualquer livre | - | [T10](https://github.com/iterate-labs-ai/caracal-1/issues/17) |

Limite Kaggle 12h por kernel. 900 steps a ~40s/step = ~10h, dentro do limite.

Schedule completo: [SCHEDULE.md](SCHEDULE.md) · Issues: [pool](https://github.com/iterate-labs-ai/caracal-1/issues)

---

## Setup (15 min, qualquer fundador)

Leia [SETUP.md](SETUP.md). TL;DR:

1. Conta Kaggle + token API (`~/.kaggle/kaggle.json`)
2. Conta GitHub adicionada na org [iterate-labs-ai](https://github.com/iterate-labs-ai)
3. Clone:
   ```bash
   git clone -b dev https://github.com/iterate-labs-ai/caracal-1.git
   cd caracal-1
   pip install -r requirements.txt
   ```
4. Smoke test local (~3 min):
   ```bash
   python scripts/smoke_test.py
   ```

---

## Workflow diario

Leia [HOWTO.md](HOWTO.md). TL;DR:

```bash
git checkout dev && git pull
git checkout -b tN-<descricao>
# ... trabalho ...
git push -u origin tN-<descricao>
gh pr create --base dev --title "tN: ..."
```

---

## Como rodar SUA sessao de treino

1. Abrir https://www.kaggle.com/code · novo notebook
2. Copiar codigo de [train/notebooks/kaggle_continued_pretrain.ipynb](train/notebooks/kaggle_continued_pretrain.ipynb)
3. Editar 5 variaveis topo:
   ```python
   SESSION = 2                                          # numero da sua sessao
   FOUNDER_HANDLE = "arturpn1"                          # seu handle Kaggle
   RESUME_DATASET = "pedroafonso2/caracal-base-3b-s01"  # output da sessao anterior
   OUTPUT_DATASET_SLUG = "caracal-base-3b-s02"
   STEPS = 900
   ```
4. Settings: GPU T4 x2 + Internet ON + Persistence
5. Save & Run All (~10h)
6. Dataset publicado automaticamente no fim
7. PR atualizando SCHEDULE.md sua linha pending -> done

---

## Como rodar EVAL

```bash
# Probe set rapido (51 prompts, mede ppl + CWE hit)
python eval/run_probe.py --adapter ./ckpt-out --out eval/reports/probe.json

# CyberGym subset
python eval/run_cybergym.py --subset slice-50 --adapter ./ckpt-out

# Baseline: Caracal vs Qwen puro
python eval/compare_baseline.py --adapter ./ckpt-out --out eval/reports/baseline.json

# Decontamination check
python eval/check_decontamination.py --strict --report data/decontamination/last-report.json
```

**Criterio v0 pass:** win_rate >= 80% no `compare_baseline` (Caracal melhor ppl que Qwen em >=80% dos probes).

---

## Arquitetura

### Modelo base

| Hiperparametro | Valor |
|---|---|
| Layers | 36 |
| Hidden | 2048 |
| Heads (Q / KV) | 16 / 2 (GQA) |
| Vocab | 151936 |
| Context | 32768 (RoPE base 1M) |
| Activation | SwiGLU |
| Norm | RMSNorm |
| Params total | 3.09B |

### LoRA adapter (Sprint 0 v0)

| Hiperparametro | Valor |
|---|---|
| r | 32 |
| alpha | 64 |
| dropout | 0.0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | ~32M (1.0%) |

### Training (v0)

| Hiperparametro | Valor |
|---|---|
| Optimizer | adamw_8bit |
| LR | 5e-5 |
| LR scheduler | cosine |
| Warmup | 3% |
| Per-device batch | 1 |
| Grad accum | 16 (batch global 16) |
| Max seq | 2048 |
| Packing | true |
| Precision | FP16 (T4 Turing nao tem BF16 nativo) |
| Grad ckpt | true (unsloth offload) |
| Steps total | 4500 (5 sessoes x 900) |

---

## Roteiro v0 -> v1 -> v2 -> v3

### v0 · Caracal Base 3B (Sprint 0, 2 semanas) · ATIVO

Continued pretrain via SFT. So Kaggle T4 x2. Pool aberto.
Output: 1 adapter LoRA mergeavel, eval probe + CyberGym subset + baseline vs Qwen.

### v1 · 5 modulos LoRA + protocolo · ate 22 jul

Modulos especialistas (todos LoRA r=32 em cima do Caracal Base):
- **Recon** (gdb embodied) - acha vulns em binaries
- **Hypothesizer** - propoe CWE + raiz da vuln
- **Crafter** (state machine) - escreve PoC
- **Validator** (SMT Z3) - prova correto
- **Patcher** (anti-suppression) - escreve fix sem suprimir

Protocolo entre eles via state machine DSL ([harness/state_machine/dsl.yaml](harness/state_machine/dsl.yaml)).

### v2 · Sandbox + RL · ate 12 ago

Sandbox Docker ([harness/sandbox/dockerfile.base](harness/sandbox/dockerfile.base)).
RL com recompensa verificavel (binary pass + anti-suppression + coverage).
DGM + MAP-Elites archive 450 cells ([archive/](archive/)).

### v3 · Recursive harness + paper · ate 31 ago

6 loops (A Pretrain · B SFT · C RL · D Scaffold · E Reward coef · F Promotion).
Kernel-D immutable ([harness/kernel/](harness/kernel/)).
Zero-day hunt + paper arXiv.

---

## Decontamination

Pipeline 3-layer + CVE-ID dedup ([eval/check_decontamination.py](eval/check_decontamination.py)):

1. **CVE-ID dedup** - blocklist em [data/decontamination/cve_blocklist.json](data/decontamination/cve_blocklist.json). Aplicada DURANTE load do dataset (em [train/continued_pretrain.py](train/continued_pretrain.py)).
2. **Layer 1 SHA256** - hash exato normalizado contra eval pool
3. **Layer 2 Jaccard 8-gram >0.3** - contra eval pool
4. **Layer 3 cosine embedding >0.85** - sentence-transformers (so se corpus <=10K)

Eval pool: CyberGym 1507 + 30 held-out CVE 2026 ([eval/held_out_2026.yaml](eval/held_out_2026.yaml)).

---

## Branches

| Branch | Pra que |
|---|---|
| `main` | Release estavel (futuro v0 merge) |
| `dev` | Default · onde tudo mergeia |
| `t1-work` ... `t12-work` | Pre-criadas por tarefa Sprint 0 |
| `tN-<descricao>` | Feature branches livres |

Branch protection nao ativada (Pro plan only) · disciplina via PR review.

---

## Milestones

- [v0 · Caracal Base 3B (Sprint 0)](https://github.com/iterate-labs-ai/caracal-1/milestone/1) · ate 8 jul
- [v1 · 5 modulos LoRA + protocolo](https://github.com/iterate-labs-ai/caracal-1/milestone/2) · ate 22 jul
- [v2 · Sandbox + RL](https://github.com/iterate-labs-ai/caracal-1/milestone/3) · ate 12 ago
- [v3 · Recursive harness + ship](https://github.com/iterate-labs-ai/caracal-1/milestone/4) · ate 31 ago

---

## Estrutura repo

```
caracal-1/
├── README.md SETUP.md HOWTO.md SCHEDULE.md CONTRIBUTING.md
├── pyproject.toml requirements.txt LICENSE
├── docs/
│   ├── plan/iterate_labs.html · plano MVP completo
│   ├── architecture.md
│   ├── recursive_harness.md   (pra v3)
│   ├── learning_trail.md
│   └── glossary.md
├── train/
│   ├── continued_pretrain.py · pipeline v0 (Unsloth + TRL + decontam)
│   ├── sft_module.py rl_grpo.py · stubs v1/v2
│   ├── configs/caracal_base_3b.yaml · hiperparams
│   └── notebooks/kaggle_continued_pretrain.ipynb · plug-and-play Kaggle
├── eval/
│   ├── probe_set.jsonl · 50 prompts ancora
│   ├── run_probe.py · ppl + CWE hit
│   ├── run_cybergym.py · subset/full bench
│   ├── compare_baseline.py · Caracal vs Qwen puro
│   ├── check_decontamination.py · 3-layer + CVE dedup
│   ├── held_out_2026.yaml · 30 CVE held-out
│   └── reports/ · output dir
├── data/
│   ├── decontamination/cve_blocklist.json · CVE bloqueados do treino
│   ├── corpus_manifest.yaml · descritor datasets
│   └── inspect_datasets.py · T3 helper
├── modules/ · 5 LoRA specialists stubs (pra v1)
├── harness/ · kernel-D, sandbox, state_machine, stop_pattern (pra v2/v3)
├── archive/ · Darwin Godel + MAP-Elites (pra v2)
├── infra/compute_inventory · slots Kaggle/Colab
├── scripts/
│   ├── smoke_test.py · valida pipeline sem treinar
│   ├── preflight.sh · check ambiente Kaggle
│   └── publish_kaggle_dataset.sh · helper publish ckpt
└── tests/
```

Um repo so. Tudo aqui.

---

## Licenca

Apache 2.0 (heranca Qwen base).

---

## Links

- Plano completo (HTML): [docs/plan/iterate_labs.html](docs/plan/iterate_labs.html)
- Plano completo (PDF): [docs/plan/iterate_labs.pdf](docs/plan/iterate_labs.pdf)
- Issues pool: https://github.com/iterate-labs-ai/caracal-1/issues
- Org: https://github.com/iterate-labs-ai
