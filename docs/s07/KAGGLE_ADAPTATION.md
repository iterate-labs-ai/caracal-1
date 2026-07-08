# Kaggle T4 x2 adaptation - benchmark suite

Kaggle limits: no docker (root sandbox), 20GB disk, 12h session, 2× T4 (16GB each), egress limited (HF/GitHub/PyPI OK).

## Suporte por bench

| Bench | Kaggle? | Adaptação |
|---|---|---|
| CTI-Bench RCM/MCQ/VSP/TAA/ATE | ✅ direct | HF `AI4Sec/cti-bench` load, gen-only inference |
| CyberMetric 500/2K/10K | ✅ direct | GitHub raw JSON download |
| SecQA v1/v2 | ✅ direct | HF `zefang-liu/secqa` |
| SecBench 44K | ✅ direct | HF `secbench-hf/SecBench` |
| MMLU computer_security | ✅ direct | HF `cais/mmlu` |
| SecEval | ✅ direct | HF `XuanwuAI/SecEval` |
| CyberSOCEval | ✅ direct | HF `facebook/CyberSOCEval` (parte da CSE4) |
| CS-Eval bilingual | ✅ direct | HF `CS-EVAL/CS-Eval` |
| CyberCertBench | ✅ direct | HF `cybercertbench/CyberCertBench` |
| CWE prediction | ✅ direct | HF `xamxte/cve-to-cwe` |
| PrimeVul | ✅ direct | HF `PrimeVul/PrimeVul` binary classify |
| **Cybench full** (40 CTF Docker) | ❌ | Requires docker + net egress. Fora do Kaggle |
| **Cybench-lite** (static subset) | ✅ adapted | `cybench_kaggle.py` - LLM recebe task_description + file_snippet, gera flag pass@1 |
| **NYU CTF full** | ❌ | Docker-based agentic. Fora Kaggle |
| **NYU CTF-lite** (static subset) | ✅ adapted | `nyu_ctf_kaggle.py` - static offline pass@1 |
| **CVE-Bench** | ❌ | Requer web target running. Fora Kaggle |
| **AIRTBench** | ❌ | Dreadnode sandbox agentic. Fora Kaggle |
| **CyberGym** | ❌ | data_dir 130GB. Fora Kaggle (dead-end conhecido) |
| **ExCyTIn-Bench** | ❌ | Azure Sentinel API required. Fora Kaggle |

**Fora do escopo Kaggle**: benches agentic com docker/network target ficam pra Modal ou cloud paga. Iterate Labs regra: **apenas Kaggle no Phase 2**.

## Cybench-lite adaptation

Cybench original (Anthropic Opus 4.7 = ~96% pass@30) precisa:
- Docker sandbox por task
- Kali Linux tools (pwntools, radare2, ghidra, etc.)
- Network egress a target service
- Interactive shell agent loop

**Cybench-lite = static offline pass@1**:
- Task = descrição + `head -c 4000` do arquivo primário embedded no prompt
- LLM só gera flag string `\boxed{flag{...}}`
- Categorias suportadas: crypto (baixar key + gerar plaintext), forensics (extrair de hexdump embedded), reverse (analisar disasm embedded), misc (raciocínio puro)
- Categorias skip: pwn (precisa remote), web (precisa target http), full agentic
- Teto realista: **~40-50%** (LLM sem tools) vs full Cybench ~96% Opus 4.7

**Dataset TBD**: `iterate-labs-ai/cybench-static` upload. Fork de `andyzorigin/cybench` filtrando por categoria + adicionando `files_snippet` (head do arquivo).

## Build datasets

Cybench-static:
```bash
git clone --recursive https://github.com/andyzorigin/cybench.git /tmp/cybench
python data/s07/build_cybench_static.py --cybench-root /tmp/cybench --out data/cybench-static.jsonl
huggingface-cli upload iterate-labs-ai/cybench-static data/cybench-static.jsonl --repo-type dataset
```

NYU CTF-lite:
```bash
git clone https://github.com/NYU-LLM-CTF/NYU_CTF_Bench.git /tmp/nyu_ctf
python data/s07/build_nyu_ctf_static.py --nyu-root /tmp/nyu_ctf --out data/nyu_ctf-static.jsonl
huggingface-cli upload iterate-labs-ai/nyu-ctf-static data/nyu_ctf-static.jsonl --repo-type dataset
```

Extractors filtram automaticamente pwn/web/binary (não roda Kaggle sem docker) e mantêm crypto/forensics/reverse/misc com file snippet head 4KB embedded no prompt.

## Estratégia comparação frontier ≥ GPT-5.4

Frontier 2026 (GPT-5.5, Opus 4.7/4.8, Sonnet 5, Gemini 3 Pro) só publicam em Cybench full + CyberGym + internal Cyber Range. **Zero rows em nossos 12 benches Kaggle-runnable**.

**3 opções**:
1. **Match frontier evals**: rodar Cybench-lite Kaggle vs full Cybench frontier (mesmo bench, diferentes settings) - honesto disclaimer
2. **Rodar frontier via API** em nossos benches (~$300-800/bench) - marketing "beat GPT-5.5"
3. **Só peer comparison** (Foundation-Sec-8B, Sec-Gemini v1)

**Recomendação**: opção 1 (cybench_kaggle static subset) + opção 3 (peer specialists) para paper honesto; opção 2 se marketing exigir.

## Notebook

`notebooks/kaggle/s07/s07_bench_kaggle.ipynb`:
- Clone repo branch `s07-hybrid-agentic`
- Install requirements
- HF login via Kaggle secret
- Download CWE XML pra hier reward
- Roda `python -m eval.s07.run_all_benches --model ... --benches ...`
- Upload JSON pra `pedroafonso2/caracal-s07-eval-<checkpoint>` (privado)

Ajusta `--n-*` args pra encaixar em 12h se time bail.
