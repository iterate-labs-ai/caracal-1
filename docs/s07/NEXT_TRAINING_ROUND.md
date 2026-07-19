# Fase 2 — RSI + RL em cybersec (plano principal)

Decisão do Pedro: a fase 2 não é continued-pretrain supervisionado. É **RL de
verdade + loop RSI** aplicado ao domínio cyber, pra mover o comportamento do
modelo, não empilhar um adapter raso.

## "Não só LoRA" — a verdade técnica sobre a T4

Em Kaggle T4 (15.9GB) **full-parameter RL de um 3B é inviável**:
- pesos fp16 ~6GB + gradientes ~6GB + optimizer states (AdamW) ~12GB = ~24GB. Estoura.
- Já brigamos com OOM só de LoRA + ref model num único 3B.

Então o mecanismo de update continua sendo LoRA — **mas o que muda vs a fase 1
não é o mecanismo, é o método**:

| Fase 1 (o que temos) | Fase 2 (o que Pedro quer) |
|---|---|
| Continued-pretrain + SFT | **GRPO (RL) com reward verificável** |
| Loss supervisionada | **Reward = acerto exato no CWE (RLVR)** |
| Treino único, adapter final | **Loop RSI: same-model propõe mutação, treina, retém se melhora** |
| Adapter descartável | **Merge no modelo a cada geração retida → modelo evolui** |

Se quiser full-parameter de verdade: precisa A100/H100 (fora do Kaggle grátis).
Aí a fase 2 roda idêntica, só troca `LoraConfig` por full-param no inner_grpo.
O código é o mesmo; muda só o hardware. Registrar como opção paga.

## Gate — não disparar treino antes de 2 números

O CÓDIGO (infra) pode ser montado agora. O TREINO (gasta GPU) espera:

1. **TPU cyber bench** com cti_bench: onde está o gap real? Se CTI-RCM < Foundation-Sec
   72-75, é ali que o RL foca (não no CyberMetric, já ~saturado a 85.8).
2. **GPU RSI Cond C (math)**: o loop gera ganho entre gerações? Se sim, escala pro
   cyber com confiança. Se `proposal_source_rate` for baixo, o outer loop virou
   busca aleatória e o redesenho vem antes de aplicar ao cyber.

## Arquitetura da fase 2 (RSI + RL cyber)

```
v_0 = melhor checkpoint cyber (s05 / caracal-base-3b-s04)
para cada geração k:
  1. v_k propõe N mutações a si mesmo (system_prompt, cot_scaffold, lr, rank)
  2. treina N LoRA candidatos via GRPO, reward = acerto CWE no CTI-RCM train
  3. avalia cada candidato no CTI-RCM dev (held-out)
  4. retém o melhor se +1pp E sem colapso (entropia, pred_distribution)
  5. merge no v_k → v_{k+1}
```

Reusa `train/ignite/C_rsi_outer.py` inteiro. O que falta é só o **reward cyber**
verificável — código montado agora (não gasta GPU, não depende de métrica):
`eval/ignite/reward.py` bench="cyber_rcm".

## Reward cyber verificável (RLVR, Shafayat-safe)

- **cyber_rcm**: extrai CWE da resposta (normalize_cwe, já corrigido) e compara
  com o gold. Match exato = 1.0. É gold verificável — não LLM-judge, imune ao
  colapso do Shafayat (2505.21444).
- **Crédito parcial hierárquico** (opcional, forte): CWE tem árvore (CWE-79 XSS é
  filho de CWE-74 injection). Acerto do pai = 0.5, irmão = 0.3. Reusa a lógica de
  eval/s07/hier_reward.py que já existe pro s07. Reward mais denso = RL aprende
  mais rápido que binário 0/1.

## Métrica nova no loop (o "e afins")

Hoje só accuracy. Fase 2 adiciona ao archive de cada geração:
- **Held-out temporal**: treina CVE <= ano X, avalia ano X+1 (nvd_feed tem data).
  Prova generalização, não memorização — crítico pra claim de paper.
- **Per-CWE breakdown**: onde erra sistemático (top-25 CWE) → guia augmentação.
- **pred_distribution** (já no primevul): pega colapso disfarçado de acerto.
- **proposal_source_rate** (já no stats.py): fração self vs fallback — prova que
  é RSI, não busca aleatória.

## Handoff pros founders (preencher slugs depois das métricas)

| Sessão | Founder | Base | Método | Métrica-alvo | Output |
|---|---|---|---|---|---|
| f2.1 | ? | caracal-base-3b-s04 | GRPO gen 0-1 | CTI-RCM held-out | ?/caracal-rsi-cyber-s01 |
| f2.2 | ? | f2.1 | GRPO gen 2-3 (resume) | CTI-RCM | ... |
| f2.3 | ? | f2.2 | GRPO gen 4-5 | CTI-RCM | ... |
| f2.4 | ? | f2.3 | ablação: RL sem RSI (só GRPO) | isola contribuição do loop | ... |
| f2.5 | ? | melhor | eval full + comparação Foundation-Sec | — | relatório |

A f2.4 (RL sem o outer loop) é o controle: separa quanto do ganho vem do RL puro
vs do loop recursivo. Sem esse controle não dá pra afirmar que o RSI ajudou.

**Regra**: nenhum founder começa sem SCHEDULE.md atualizado e sem confirmar que a
sessão anterior publicou. Mesmo protocolo do PROXIMA_SESSAO.md.
