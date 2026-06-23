# KERNEL-D · arquivos intocaveis pelo loop

Estes arquivos sao a fronteira de seguranca. So humanos editam.

## Regras

1. Qualquer mudanca em arquivo deste diretorio exige:
   - 2 revisores obrigatorios (CODEOWNERS + workflow guard)
   - ADR registrado em `harness/decisions/NNNN-titulo.md`
   - Validacao: suite eval passa + probes anti-hack + held-out 30 CVE nao tocado

2. O loop (Loops C, D, E) NAO pode editar:
   - `classes.yaml`
   - `budgets.yaml`
   - `reward_registry.yaml` (funcao oracle)
   - `_hashes.json`
   - `../sandbox/*` (config sandbox)
   - `../../eval/held_out_2026.yaml`

3. O loop PODE editar coeficientes de shaping via Loop E:
   - `reward_registry.yaml::shaping_coefficients_v1` (com sanity check)

4. STOP-pattern detector roda em cada gate. Trigger = trajetoria descartada + flag em `../incidents/`.

## Referencias

- Anthropic ASL tiers (RSP v3)
- STOP paper · arXiv 2310.02304
- Iterate Labs harness recursivo (docs/recursive_harness.md)
