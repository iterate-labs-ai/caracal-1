# Sandbox · Caixa de areia hardened

Containers Docker em Modal CPU. gVisor + firejail + nsjail + seccomp + network off.

Veja `docs/architecture.md` e Issue C7.

## Attack tests obrigatorios

5 testes que TODOS devem bloquear:

1. Tentar acessar internet via curl/wget
2. Tentar escrever fora de /workdir
3. Tentar sudo
4. Tentar fork bomb
5. Tentar mount

Status: bloqueado / OK.

## Limites

- timeout 60 segundos por execucao
- memoria 2 GB
- pids 512
- rede off (allowlist explicita por task se necessario)
- /workdir tmpfs (efemero)
- cap-drop ALL
- no-new-privileges
