# Decontamination pipeline · 3 camadas

Filtro garantindo zero overlap entre corpus de treino e:
- CyberGym 1507
- Held-out 30 CVE 2026

## Camadas

1. **SHA256 hash exato** · snippet hash match
2. **Jaccard 8-gram** · threshold 0.3
3. **Cosseno embedding** · threshold 0.85 (modelo: sentence-transformers/all-MiniLM-L6-v2)

Adicional: **CVE-ID dedup** · treino nunca tem CVE que vai entrar em avaliacao.

## Implementacao

Veja `eval/check_decontamination.py` (task C6).

## Audit

PRs adicionando dados ao corpus disparam `.github/workflows/decontamination_check.yml`. Falsos positivos < 2% em amostra manual de 100.
