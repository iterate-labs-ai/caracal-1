# modules · scaffolds para v1

Cada subdir contem config.yaml + dataset.yaml + system_prompt.md (quando tiver) para um modulo especialista LoRA.

Estado atual: **scaffolds**. Configs apontam para Qwen base. Treino real comeca apos v0 fechar.

Pipeline v1 esperado:
1. Substituir `base_model` em cada config.yaml pelo Kaggle Dataset final do v0 (`aletlucas/caracal-base-3b-v0`)
2. Implementar `train/sft_module.py` (sera reescrito)
3. Treinar 5 LoRA adapters separados sobre o Caracal Base

Modulos:

| Modulo | Dominio | Status |
|---|---|---|
| recon | acha vulns em binarios (gdb embodied) | planejado v1 |
| hypothesizer | propoe CWE + raiz | planejado v1 |
| crafter | escreve PoC via state machine | planejado v1 |
| validator | prova correto via SMT Z3 | planejado v1 |
| patcher | escreve fix sem suprimir | planejado v1 |

Configs aqui sao referencia. Cada modulo vai precisar dataset proprio + sft_module.py funcional antes de treinar.
