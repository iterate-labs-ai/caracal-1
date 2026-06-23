# data/

Datasets passam por pipeline de decontaminacao antes de entrar no treino. Veja `eval/check_decontamination.py`.

## Manifesto

`corpus_manifest.yaml` lista todas as fontes para Caracal Base continued pretrain. Total 1.7B tokens.

## Estrutura local (gitignore)

```
data/
├── corpus_manifest.yaml  (commitado)
├── decontamination/      (commitado · regras + reports)
├── raw/                  (gitignored · download local)
├── processed/            (gitignored · pos-pipeline)
└── corpus/               (gitignored · streamed)
```

## SFT estagio 2

Pasta separada por modulo. Veja `modules/<nome>/README.md` para datasets especificos.

## Avaliacao

`eval/held_out_2026.yaml` (KERNEL-D) tem 30 CVE held-out 2026. NUNCA tocar para treino.
