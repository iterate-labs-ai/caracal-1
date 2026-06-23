# Arquitetura Caracal-1

## Visao geral

Caracal-1 e Transformer decoder-only de 3.09 bilhoes de parametros. Base Qwen2.5-Coder-3B-Instruct. 36 camadas. Atencao agrupada 16 cabecas Q por 2 cabecas KV. SwiGLU MLP. RMSNorm. RoPE base 1.000.000.

## Camadas

```
Texto -> Tokenizador BPE 151.936 vocab
      -> Camada embedding (token -> 2048)
      -> 36 blocos decoder (norm, atencao, residual, norm, MLP, residual)
      -> RMSNorm final
      -> Cabeca de linguagem (2048 -> 151.936 logits)
      -> Proximo token
```

## Caracal Base 3B (fundacional)

= Qwen2.5-Coder-3B + 1.5B tokens continued pretrain cyber + 200M tokens traces de execucao.

Adicionalmente treinado:
- AST attention overlay head (+12M parametros)
- Mental Simulator head (+50M parametros)
- Inline verifier head (+30M parametros · PRM)

Total backbone: ~3.18 bilhoes parametros, congelado durante treino de modulo.

## 5 Modulos LoRA especialistas

Cada modulo e LoRA r=32 alpha=64 sobre Caracal Base. Aplica em 7 modulos por bloco × 36 blocos = 252 adapters, totalizando ~33 milhoes de parametros (1.1% do base).

1. **Recon** · scout + embodied gdb live
2. **Hypothesizer** · 3 hipoteses de bug class por target
3. **Crafter** · state machine policy + PoC
4. **Validator** · spec formal + Z3 bounded
5. **Patcher** · unified diff + anti-supressao

Cada adapter ~130MB em disco. 5 adapters totais = 650MB de novidade sobre 6GB do base.

Veja `modules/<nome>/README.md` para detalhes especificos.
