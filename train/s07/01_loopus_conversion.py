"""Caracal s07.B - LoopUS post-training conversion to looped LLM.

Kevin session 2/7. ~12h Kaggle T4 x2.

Stage: L1 (latent recursion).

Paper: LoopUS arxiv 2605.11011 + Retrofitted Recurrence arxiv 2511.07384.

Flow:
1. Load caracal-s07-base-tokens from s07.A
2. Decompose Qwen layers:
   - Encoder block: layers 0-9 (10 layers)
   - LOOP block: layers 10-21 (12 layers, weight-tied iter <=8)
   - Decoder block: layers 22-31 (10 layers Qwen 32-layer)
3. Add Retrofitted Recurrence LoRA r=32 nas LOOP layers (weight sharing)
4. Adaptive halt: linear head sobre LOOP output -> halt prob
5. SFT em 10K xamxte/cve-to-cwe subset com recursion-aware loss
6. Save adapter HF: pedroafonso2/caracal-s07-loopus
"""

# TODO Kevin:
# - implement LoopUS decomposition (custom forward pass com weight reuse)
# - add LoRA on LOOP block only
# - implement halt head
# - SFT loss = next-token + halt regularization
# - smoke test recursion (3-5 iter typical)
# - eval CTI-RCM subset 150 samples

ENCODER_LAYERS = (0, 10)  # 0-9 inclusive
LOOP_LAYERS = (10, 22)  # 10-21 inclusive, weight-tied
DECODER_LAYERS = (22, 32)  # 22-31 inclusive
MAX_LOOP_ITER = 8


def main():
    raise NotImplementedError("s07.B scaffolding - implement after s07.A ship")


if __name__ == "__main__":
    main()
