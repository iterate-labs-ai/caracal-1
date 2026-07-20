> **RETRATAÇÃO 2026-07-20.** O head-to-head contra o modelo base derrubou a
> claim principal deste documento. Ver "Caracal vs base" abaixo antes de usar
> qualquer número daqui. Resumo: o adapter s04 **piora** o CTI-RCM (-4.6pp) e o
> CyberMetric 85.8% é essencialmente o que o Qwen2.5-Coder-3B base já faz
> sozinho (84.2%). O texto original fica preservado abaixo como registro.

# Caracal vs base — head-to-head medido (2026-07-20, GPU T4, n na tabela)

Mesmo harness, mesmo prompt, mesma máquina. `Caracal` = Qwen2.5-Coder-3B-Instruct
+ adapter `caracal-base-3b-s04`. `base` = o mesmo Qwen sem adapter.

| bench | Caracal | base | delta | n |
|---|---|---|---|---|
| cti_bench.rcm | 39.8% | **44.4%** | **-4.6pp** | 500 |
| cti_bench.mcq | 44.2% | **50.6%** | **-6.4pp** | 500 |
| cybermetric | 85.8% | 84.2% | +1.6pp | 500 |
| secqa v1 | 99.1% | 99.1% | 0.0pp | 110 |
| secqa v2 | 97.0% | 98.0% | -1.0pp | 100 |
| secbench | **79.7%** | 74.0% | **+5.7pp** | 300 |
| mmlu_security | 75.0% | 73.0% | +2.0pp | 100 |
| seceval | 42.5% | 40.5% | +2.0pp | 200 |
| cybersoceval.malware | 14.0% | 20.0% | -6.0pp | 100 |
| cybersoceval.threat_intel | 17.0% | 22.0% | -5.0pp | 100 |

**Leitura honesta:**

1. **O fine-tune não entregou specialist.** Na claim que importa (CTI-RCM, o
   benchmark comparável com Foundation-Sec-8B), o adapter é **pior que o base**.
2. **A claim do CyberMetric era do Qwen, não nossa.** 84.2% do base vs 85.8% com
   adapter: +1.6pp em n=500 é ruído (CI ±~2pp). Posicionar "Caracal-3B na faixa
   do Llama-3.1-8B" era atribuir ao adapter um mérito do modelo base.
3. **Único ganho acima de ruído**: secbench +5.7pp (n=300).
4. **Distância real do specialist**: CTI-RCM 39.8% (ou 44.4% do base) contra
   Foundation-Sec-8B 72-75%. Headroom de ~28pp.

**Consequência pra fase 2 (RSI+RL):** o v0 do loop passa a ser o **modelo base**,
não o adapter s04 — começar do adapter seria começar 4.6pp abaixo no alvo. O
objetivo do RL deixa de ser "melhorar mais" e passa a ser subir de 44.4% em
direção a 72-75. Piso de colapso medido no dev: responder sempre CWE-79 = 27.3%.

---

# (histórico, superado pela retratação acima) Onde o Caracal-3B está no mundo — CyberMetric-500 side-by-side

Única comparação estatisticamente válida (set fixo de 500). Generalistas open
source marcados com [G], specialists cyber com [C].

```
Llama-3.1-70B-Instruct   [G] 70B  93.0
GPT-4o-mini              [fechado]  88.9   (referencia)
DeepHat-v1-7B            [C]  7B   86.9
Qwen2.5-7B-Instruct      [G]  7B   85.9
>>> CARACAL-3B (nos)     [C]  3B   85.8 <<<
Primus-Base              [C]  8B   85.4-86.6
Llama-3.1-8B-Instruct    [G]  8B   84.7-85.6
Foundation-Sec-8B        [C]  8B   84.8
Gemma-3-4B-it            [G]  4B   76.8
```

**Leitura**: o Caracal-3B empata com os generalistas open source de 7-8B
(Qwen2.5-7B 85.9, Llama-3.1-8B 84.7-85.6) usando **2.3-2.7x menos parametros**,
e empata com os specialists cyber de 8B (Foundation-Sec 84.8, Primus 85.4). Bate
o Gemma-3-4B por +9pp. Fica a ~1pp do teto da faixa pequena (DeepHat-7B 86.9)
com metade dos params. Caveat: harness proprio, nao controlado — posicionamento
aproximado, nao ranking rigoroso.

## TPU: descartado com evidencia (2026-07-19)

Run `caracal-bench-tpu-s07` na TPU v3-8 mediu:

```
modelo na TPU em 65s
q1: 9538.4s                                    <- 2h39 so de compilacao XLA
primeira: 9538.4s | media das ultimas 5: 21.6s
```

Mesmo depois de aquecer, **21.6s/questao** (GPU faz ~1-2s). 500 questoes = 3h
por bench; 7 benches nao cabem nas 20h. O run morreu de DeadKernel no cti_bench.
O padding pra bucket fixo reduziu o numero de shapes mas nao salvou o custo.

**Conclusao: nao rodar bench em TPU pra esse stack.** O notebook TPU agora tem
gate DURO (`raise SystemExit` se warm > 10s/questao) em vez de so avisar - antes
ele imprimia "LENTO DEMAIS" e seguia mesmo assim, queimando o run inteiro.
Bench cyber vai pra GPU.

**Segunda claim a habilitar**: CTI-RCM (CVE->CWE) verdadeiro ja esta implementado
em `eval/s07/benches/cti_bench.py` (n=1000, prompt oficial embutido no dataset,
normalize_cwe corrigido). Faltou so entrar na lista de benches do run. Rodando,
compara direto contra Foundation-Sec-8B RCM 72-75, o teto real do specialist.

---

# Baselines cyber open-weight — fontes primárias verificadas

Cada número foi lido direto na fonte primária (leaderboard oficial, PDF do autor
do modelo ou do benchmark). Nada de agregador de terceiro. Onde duas fontes
discordam, ambas ficam registradas — não se faz média.

Fontes (todas abertas e lidas):
- **[FSb]** Cisco Foundation-Sec-8B — arxiv.org/abs/2504.21039
- **[FSi]** Cisco Foundation-Sec-8B-Instruct — arxiv.org/pdf/2508.01059
- **[FScard]** huggingface.co/fdtn-ai/Foundation-Sec-8B
- **[PB]** huggingface.co/trendmicro-ailab/Llama-Primus-Base
- **[SEl]** xuanwuai.github.io/SecEval/leaderboard.html
- **[CTIp]** arxiv.org/abs/2406.07599 · **[SB]** arxiv.org/pdf/2412.20787
- **[SOC]** arxiv.org/abs/2509.20166 · **[Q3]** arxiv.org/abs/2505.09388

## CyberMetric-500 (set fixo de 500 MCQ human-validated)

Único bench sem problema de n: o set é totalmente enumerado.

| modelo | params | score | fonte |
|---|---|---|---|
| Llama-3.1-70B-Instruct | 70B | 0.930 | [FSi] |
| GPT-4o-mini | — | 0.889 | [FSi] |
| DeepHat-v1-7B | 7B | 0.869 | [FSi] |
| Primus-Base | 8B | 0.854 / 0.866 | [FSi]/[PB] |
| Qwen2.5-7B-Instruct | 7B | 0.859 | [FSi] |
| Llama-3.1-8B-Instruct | 8B | 0.847 / 0.856 | [FSi]/[PB] |
| Foundation-Sec-8B (base) | 8B | 0.848 | [FSb] |
| Foundation-Sec-8B-Instruct | 8B | 0.830 | [FSi] |
| Gemma-3-4B-it | 4B | 0.768 | [FSi] |
| **Caracal-3B (nós)** | **3B** | **0.858** | nosso harness |

**Leitura**: 85.8% coloca o Caracal-3B na faixa do Llama-3.1-8B-Instruct
(84.7-85.6), acima do Gemma-3-4B (76.8), com **menos da metade dos parâmetros**
do peer de 8B. Caveat: nosso harness (formato de prompt + parse XML) difere do
da Cisco/Trend — não é uma comparação controlada.

## CTI-Bench RCM (CVE→CWE) e MCQ

| modelo | params | MCQ | RCM | fonte |
|---|---|---|---|---|
| Foundation-Sec-8B (card) | 8B | 67.39 | 75.26 | [FScard] |
| Foundation-Sec-8B (base) | 8B | 66.2 | 72.0 | [FSb] |
| Foundation-Sec-8B-Instruct | 8B | 64.4 | 69.2 | [FSi] |
| DeepHat-v1-7B | 7B | 64.5 | 66.4 | [FSi] |
| Primus-Base | 8B | 66.8 | 67.8 | [PB] |
| Qwen2.5-7B-Instruct | 7B | 64.4 | 57.2 | [FSi] |
| Llama-3.1-8B-Instruct | 8B | 61.7 | 55.8 | [FSi] |
| Llama-3.1-70B-Instruct | 70B | 69.5 | 62.3 | [FSi] |

Nosso `cwe_prediction` (54%, n=50) usa o dataset xamxte/cve-to-cwe, que **não é
o CTI-RCM** — não é comparável diretamente. O RCM verdadeiro é o alvo a bater;
Foundation-Sec-8B RCM 72-75 é o teto do specialist de 8B.

## SecEval — harness incompatível entre fontes

| modelo | params | score | fonte |
|---|---|---|---|
| gpt-4-turbo | — | 79.07 | [SEl] leaderboard oficial 2023 |
| Foundation-Sec-8B-Instruct | 8B | 83.3 | [FSi] |
| Llama-3.1-8B-Instruct | 8B | **49.7 / 85.5** | **[PB] vs [FSi]** |
| Primus-Base | 8B | **50.1 / 84.1** | **[PB] vs [FSi]** |
| Mistral-7B-v0.1 | 7B | 43.65 | [SEl] |

**SecEval não é comparável cruzando fontes**: pro mesmo Llama-3.1-8B a Trend
reporta 49.7 e a Cisco 85.5. O leaderboard oficial topa em 79. Harness diferente.
Nosso 40% (n=50) não pode ser posicionado.

## Benches sem fonte primária publicada

- **SecQA v1/v2**: nenhum modelo open moderno publicou. A Cisco lista como
  "revisado, não usado". Nosso 99.1/97.0 fica **standalone, sem ranking** — e
  valores perto do teto sugerem risco de saturação/contaminação.
- **MMLU computer_security**: nenhuma fonte primária quebra por subset pros
  modelos-alvo (só MMLU full). Nosso 80.0 (n=20) não posiciona.
- **CyberSOCEval**: paper [SOC] só reporta em gráfico de barra, sem tabela
  numérica. Malware-analysis top ≈ 23-34%.
- **PrimeVul**: nenhum baseline publicado nos modelos-alvo.
- **Qwen3 tech report [Q3]** e HF cards de Qwen2.5/Qwen3/WhiteRabbitNeo/DeepHat/
  Lily: **zero benches cyber reportados pelos autores**.

## Discrepâncias registradas

1. SecEval Llama-3.1-8B: 0.497 [PB] vs 0.855 [FSi] — gap de 0.35, harness distinto.
2. Foundation-Sec-8B CTI: card 67.39/75.26 vs report 66.2/72.0 — RCM difere ~3pp.
3. CyberMetric-500: Llama-3.1-8B 0.847 [FSi] vs 0.856 [PB] — ~1pp, harness distinto.

## Veredito — o que dá pra afirmar

- **Defensável (1 claim)**: CyberMetric-500 = 85.8%, na faixa do Llama-3.1-8B com
  3B params. Set fixo, sem problema de n. Sempre com o caveat do harness.
- **Não defensável**: SecEval (n=50), MMLU-sec (n=20), CWE-pred/PrimeVul (n=50).
  n=20-50 dá CI de ±12-22pp — estatisticamente sem significado contra benches de
  100-2000 questões.
- **Sem posição**: SecQA 99.1/97.0 — nenhum baseline comparável publicado.

**Conclusão pro paper/README**: publicar só a comparação CyberMetric-500 como
claim peer-relativo. Apresentar SecQA/MMLU-sec/SecEval como diagnóstico interno,
com n explícito e flag "não comparável". Pra reivindicar CTI-RCM como specialist,
rodar o CTI-RCM verdadeiro (AI4Sec/cti-bench), não o xamxte, com n completo (1000).
