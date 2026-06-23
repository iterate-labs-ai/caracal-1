# Archive · Darwin Godel + MAP-Elites

Arquivo evolutivo dos checkpoints. Preserva diversidade.

## Estrutura

- 450 celulas MAP-Elites (30 CWE × 5 primitives × 3 langs)
- Dentro de cada celula: parent sampling Darwin Godel formula
- Capacidade 32 checkpoints top por celula, prune dominados weekly
- Snapshot cada 500 passos RL
- Triplet evolutivo: (checkpoint, harness_config, primitive_library)

## Primitive library

Banco vetorial Qdrant em `primitive_library/`. Modelo escreve quando descobre primitive nova (Loop D scaffold mutation). Cresce ao longo dos sprints.

## Referencias

- Sakana DGM · arXiv 2505.22954
- MAP-Elites · arXiv 1504.04909
