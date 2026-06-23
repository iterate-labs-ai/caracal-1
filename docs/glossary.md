# Glossario

| Termo | Significado |
|---|---|
| LoRA | Low-rank adapter. Treina matrizes pequenas. r=32 = posto 32. ~1% dos parametros do total. |
| QLoRA | LoRA quantizado. Base 4-bit + adapter 16-bit. Cabe 7B em 16GB. |
| SFT | Supervised Fine-Tuning. Treino com pares (input, output). |
| Continued pretrain | Mid-training. Continua treino com corpus novo antes do SFT. |
| GRPO | Group Relative Policy Optimization. Algoritmo RL DeepSeek-R1. |
| DAPO | GRPO 2026 com clip assimetrico eps_low=0.2 eps_high=0.28. |
| Reward verificavel | Recompensa binaria executavel. Imune a reward hacking. |
| PoC | Proof of Concept. Codigo curto que reproduz vulnerabilidade. |
| CVE | Common Vulnerabilities and Exposures. Catalogo publico. |
| CWE | Common Weakness Enumeration. Classificacao de tipos de bug. |
| ASan/UBSan/MSan | Sanitizers gcc/clang. Detectam OOB, UB, mem nao init. |
| gdb/mi | Machine Interface do gdb. Output estruturado parseable. |
| gcov | Coverage tool do gcc. |
| OSS-Fuzz | Servico Google fuzzing continuo em OSS critico. |
| Darwin Godel Machine | Arquivo evolutivo checkpoints. Sakana 2025. |
| MAP-Elites | Quality-diversity. Guarda melhor por combinacao de comportamentos. |
| vLLM | Motor inferencia com PagedAttention. |
| Counterfactual | Mutacao minimal que NAO triggaria bug. Sinal causal. |
| SMT | Satisfiability Modulo Theories. Z3 da Microsoft. |
| SAE | Sparse Autoencoder. Decompoe ativacoes em features interpretaveis. |
| CyberGym | Benchmark Berkeley ICLR 2026. 1507 vulns. |
| HuggingFace Hub | Plataforma pra publicar modelos. Equivalente GitHub pra pesos. |
| KERNEL-D | Componentes intocaveis pelo loop. So humanos editam. |
