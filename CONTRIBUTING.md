# CONTRIBUTING

## Branches

- `main` · release estavel · PR-only · 2 revisores
- `dev` · branch default de trabalho · features mergeam aqui primeiro
- `exp/<nome>` · branches experimentais por fundador
- `release/v1` · snapshot pre-ship (sera criada Sprint 4)

## Workflow

1. Issue existe e voce claim
2. `git checkout -b cN-curta-descricao dev`
3. Commits atomicos com mensagens claras (imperativo: "add X", "fix Y")
4. Push: `git push -u origin cN-curta-descricao`
5. PR contra `dev` linkando issue: `Closes #N`
6. Pedir review de fundador que NAO escreveu o codigo
7. Issues com tag `kernel-D` (C5, C6, C8): 2 revisores obrigatorios + ADR registrado
8. CI verde obrigatorio (lint + tests + decontamination check)
9. Merge via squash
10. Apos merge: comentar `done` na issue + fechar

## Estilo de codigo

- Python 3.11+
- Ruff format + lint (config em pyproject.toml)
- Type hints onde fizer sentido
- Docstrings em funcoes publicas (estilo Google)
- Pytest pra testes (tests/test_*.py)

## Estilo de commit

- Imperativo presente: "add", "fix", "update"
- Primeira linha <72 chars
- Corpo opcional explicando o porque
- Footer: `Closes #N` se aplicavel

Exemplos bons:
```
add reward function verifier-gated v0

implements binary reward with anti-suppression check.
gcov coverage signal for dense reward shaping.

Closes #8
```

```
fix sandbox SIGTERM handler losing final checkpoint

Closes #42
```

## Kernel-D (intocavel pelo loop)

Arquivos abaixo so podem ser editados por humanos (2 revisores + ADR):

- `eval/held_out_2026.yaml`
- `harness/kernel/classes.yaml`
- `harness/kernel/budgets.yaml`
- `harness/kernel/reward_registry.yaml`
- `harness/kernel/_hashes.json`
- `harness/sandbox/*` (config)

CODEOWNERS faz enforcement no GitHub. Workflow `.github/workflows/kernel_d_guard.yml` valida.

## Decontamination

Antes de merge em qualquer PR que adiciona dado a `data/` ou modifica corpus, CI roda `eval/check_decontamination.py` que valida 3-camadas (hash + 8-gram + cosseno embedding) contra CyberGym 1507 e held-out 30 CVE 2026. Falha = PR red.

## Test

Workflow `.github/workflows/ci.yml` roda:
- ruff format + lint
- pytest unit tests
- decontamination check em PRs que mexem em data/
- KERNEL-D guard em PRs que mexem em harness/kernel/ ou eval/held_out_2026.yaml
