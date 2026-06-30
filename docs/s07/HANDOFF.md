# Caracal s07 - Founder Relay Handoff

7 sessions x 12h Kaggle T4 x2 = 84h total.

## Session breakdown

| Sess | Founder | Goal | Notebook | Deliverable HF |
|---|---|---|---|---|
| s07.A | Pedro | Setup tokenizer + base | s07A_setup.ipynb | caracal-s07-base-tokens |
| s07.B | Kevin | LoopUS + Retrofitted Recurrence | s07B_loopus.ipynb | caracal-s07-loopus |
| s07.C | Arthur | Reasoning tokens SFT (Fast Quiet-STaR) | s07C_reasoning.ipynb | caracal-s07-reasoning |
| s07.D | Vitor | Tool calling SFT (Hammer + xLAM) | s07D_tools.ipynb | caracal-s07-tools |
| s07.E | Alexandre | Minerva RLVR (GRPO + ACR) | s07E_minerva_rlvr.ipynb | caracal-s07-rl |
| s07.F | Pedro | RSD + DGM mutation | s07F_rsd_dgm.ipynb | caracal-s07-v2 |
| s07.G | Kevin | Full eval + paper draft | s07G_eval_paper.ipynb | paper + report |

## Handoff template per session

```markdown
# Caracal s07.X Handoff Report

**From**: [founder name]
**To**: [next founder]
**Date**: [ISO]
**Kernel**: pedroafonso2/caracal-s07.X-...

## Stage completed
- LoRA adapter saved: [HF path]
- Trainer state: [path/trainer_state.json]
- Steps: 0 -> N

## Eval delta vs previous
- CTI-RCM: X% (Δ +Ypp)
- CyberMetric: X% (Δ +Ypp)
- McNemar p-value: 0.0XX

## Anomalies / failures
- [list]

## Next session start command
```bash
kaggle kernels pull -p . pedroafonso2/caracal-s07.[X+1]-...
```

## Files modified
- [list]

## TODO for next founder
- [explicit task list]
```

## Sequencing dependencies

```
s07.A (setup) --> s07.B (loopus) --> s07.C (reasoning)
                                            |
                                            v
                  s07.E (RLVR) <-- s07.D (tools)
                       |
                       v
                  s07.F (RSD+DGM)
                       |
                       v
                  s07.G (eval+paper)
```

## Failure recovery

- Session killed mid-train: trainer auto-saves every 100 steps, next founder resumes from latest checkpoint
- OOM: reduce per_device_batch by 25%, increase gradient_accumulation
- Kernel cancelled: cada session deve push checkpoint a cada 1h via `kaggle datasets version`
- Loss spike: revert one checkpoint, continue from previous

## Pre-requisitos antes de s07.A

- [ ] Caracal s05 ship (Alexandre finalizando)
- [ ] Branch s07-hybrid-agentic merged em dev
- [ ] 5 founder Kaggle accounts configurados
- [ ] HF org access pra pedroafonso2/* uploads
- [ ] NVD API key (free, https://nvd.nist.gov/developers)
- [ ] Anthropic API key Pedro (Sonnet 4.6 batch teacher gen)
