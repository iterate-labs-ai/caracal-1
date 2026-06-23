# Notebooks template

Pronto-pra-rodar em Kaggle ou Colab Pro+. Copia, ajusta as 5 linhas iniciais (slot info), executa todas as celulas.

## kaggle_continued_pretrain.ipynb

- Auth: Kaggle Secrets (HF_TOKEN, WANDB_API_KEY)
- Hardware: T4 (gratis com phone-verified)
- Tempo: ~9h por 5500 passos
- Output: HF Hub revision tagged

## colab_continued_pretrain.ipynb

- Auth: Colab Secrets sidebar
- Hardware: T4 ou A100 (Pro+ esporadico)
- Tempo: ~9h T4 ou ~4h A100
- Output: HF Hub revision tagged

## Como usar

1. Pegar slot livre em SCHEDULE.md (PR pequena)
2. Abrir notebook no Kaggle/Colab
3. Editar primeira celula: SESSION_NUMBER, RESUME_REVISION, OUTPUT_REVISION
4. Configurar Secrets (HF_TOKEN + WANDB_API_KEY)
5. Run all
6. Apos terminar: PR atualizando SCHEDULE.md (pending -> done com revision)
7. Sinalizar Discord #caracal-relay
