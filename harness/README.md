# harness · scaffolds para v2 / v3

Estado atual: **scaffolds**. Codigo real comeca apos v1 (5 modulos LoRA) fechar.

Estrutura:

| Subdir | Pra que | Status |
|---|---|---|
| kernel/ | classes.yaml + budgets.yaml + reward_registry.yaml (kernel-D immutable) | scaffold v3 |
| sandbox/ | dockerfile.base para sandbox RL | scaffold v2 |
| state_machine/ | dsl.yaml para protocolo entre modulos | scaffold v1/v2 |
| decisions/ | registro de decisoes do harness | scaffold v3 |
| incidents/ | postmortems de runs | scaffold v3 |
| schemas/ | tipos compartilhados | scaffold v3 |
| stop_pattern_detector.py | detecta loops e degenerate behavior | scaffold v3 |

v0 nao usa nada disso. v1 talvez encoste em state_machine/dsl.yaml.
v2 sandbox + reward.
v3 kernel-D + recursive harness.
