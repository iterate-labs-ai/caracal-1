# archive · scaffold para v2

Darwin Godel + MAP-Elites archive para coletar e amostrar parents durante RL.

Estado atual: **scaffold**. Codigo real comeca apos v1 fechar.

## Arquivos

| Arquivo | Pra que | Status |
|---|---|---|
| schema.py | tipos do archive | scaffold v2 |
| dgm_sampler.py | sampler Darwin Godel | scaffold v2 |
| map_elites.py | MAP-Elites 450 cells (30 CWE × 5 primitives × 3 langs) | scaffold v2 |
| primitive_library/ | indices Qdrant de primitivas | scaffold v2 |

## Estrutura prevista (v2)

- 450 celulas MAP-Elites
- Dentro de cada celula: parent sampling Darwin Godel
- Capacidade 32 checkpoints top por celula, prune dominados weekly
- Snapshot cada 500 passos RL
- Triplet evolutivo: (checkpoint, harness_config, primitive_library)

## Primitive library

Banco vetorial Qdrant em `primitive_library/`. Modelo escreve quando descobre primitive nova (Loop D scaffold mutation). Cresce ao longo dos sprints.

## Referencias

- Sakana DGM · arXiv 2505.22954
- MAP-Elites · arXiv 1504.04909
