# Modulo recon

LoRA r=32 alpha=64 sobre Caracal Base 3B. Treina apenas adapters (130MB).

## Status

Adapter ainda nao publicado. Aguardando Caracal Base finalizar (C4) e tarefa de SFT especifica.

## Configuracao

Veja `config.yaml` neste diretorio.

## Treino

```bash
python train/sft_module.py --module recon --config modules/recon/config.yaml
```

## Pesos finais

Quando publicado: `iterate-labs/caracal-1-recon-lora` no HuggingFace Hub.
