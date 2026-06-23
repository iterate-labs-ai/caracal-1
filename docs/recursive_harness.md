# Harness recursivo · 6 loops self-improving

Caracal-Ω-Recursive opera em 6 loops aninhados. Boundary intocavel: KERNEL-D (evaluator, sandbox, reward registry, budget enforcer, classes config). So humanos editam kernel-D.

## Os 6 loops

| Loop | Muta | Quem aprova | Frequencia | Validador |
|---|---|---|---|---|
| **A · Pretrain** | Pesos Caracal Base | 2 humanos PR | 1 vez Sprint 1 | Probe + HumanEval+ regression |
| **B · SFT** | Pesos LoRA modulo | 1 humano PR | 1 vez/modulo Sprint 2 | Probe + CyberGym slice 50 |
| **C · RL** | Pesos LoRA via GRPO/DAPO | auto + gate cada 500 passos | continuo Sprint 3 | Eval suite + probes anti-hack |
| **D · Scaffold** | prompts, tool budget, state machine | auto + Pareto dev slice | continuo apos 1000 passos | Dev slice 100 prompts (NAO CyberGym) |
| **E · Reward coef** | α β γ δ ε | auto + sanity probes | continuo apos 1500 passos | Coverage + length + canary probes |
| **F · Promocao kernel** | mutacoes Loop D/E consistentes 3+ vezes | 2 humanos + ADR | weekly review | Manual auditoria |

## Boundaries

| Componente | Loop pode mutar? | Por que |
|---|---|---|
| Evaluator suite | NUNCA | Sistema nao pode editar proprio teste (STOP precedent) |
| Reward function (oracle) | NUNCA | Gameabilidade infinita |
| Reward shaping coef α β γ δ ε | SIM (Loop E) | Coeficientes auxiliares, funcao oracle preservada |
| Sandbox config | NUNCA | Loop nao reduz proprio isolamento |
| Budget enforcer | NUNCA | STOP: sistema tentou remover proprio budget |
| Classes A/B/C/D | NUNCA | Loop nao escala privilegio |
| Prompts e scaffolds | SIM (Loop D) | Classe A, edita livre |
| Tools e retrieval | SIM (Loop D) | Classe B, 1 review |
| Pesos do modelo | SIM (Loops A/B/C) | Classe C, 2 reviewers + 3 seeds + suite |

## STOP-pattern detector

Probe automatica em cada gate detecta tentativas do loop de:

- Editar arquivos kernel-D
- Aumentar proprio budget
- Reduzir proprio isolamento
- Suprimir crashes em vez de patchar
- Coletar canary strings de corpus de treino

Trigger = trajetoria descartada + flag em `harness/incidents/YYYY-MM-DD.md` + review humano.

## Referencias

- Darwin Godel Machine (Sakana 2025) · arXiv 2505.22954
- SICA · 17 to 53 percent SWE Bench Verified
- STOP paper · arXiv 2310.02304
- SEAL · NeurIPS 2025
