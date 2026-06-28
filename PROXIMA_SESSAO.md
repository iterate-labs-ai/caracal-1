# Pegar a proxima sessao · quickstart pra founder

Quando voce vai entrar pro relay (sessao 2, 3, 4 ou 5).

**Cenario (06-28):** Pedro s01, Kevin s02, Arthur s03 ja rodaram. Agora **Vitor (s04)** entra. Mesmo passo a passo vale pra Alexandre depois.

---

## 0. Pre-requisito (1 minuto)

Antes de qualquer coisa, confirme que a sessao anterior **realmente terminou**:

```bash
kaggle datasets list -s "caracal-base-3b"
```

Deve listar `pedroafonso2/caracal-base-3b-s01` (ou seja qual for a anterior). Se nao listou, ainda nao acabou. Espera.

Tambem confere no [SCHEDULE.md](SCHEDULE.md): linha da sessao anterior tem que estar `done`.

---

## 1. Abrir notebook Kaggle (2 minutos)

1. Vai em https://www.kaggle.com/code
2. Clica **"+ New Notebook"**
3. Cola conteudo de [train/notebooks/kaggle_continued_pretrain.ipynb](train/notebooks/kaggle_continued_pretrain.ipynb) cell por cell
   (ou faz **File -> Import Notebook** apontando pro arquivo do repo)

---

## 2. Configurar GPU + Internet (30 segundos)

Na sidebar direita (clica **"Notebook options"** se estiver fechada):

- **Accelerator:** GPU T4 x2
- **Internet:** ON
- **Persistence:** Variables and Files (ja vem assim)

---

## 3. Editar 5 variaveis no topo da primeira cell (1 minuto)

Esta e a UNICA cell que voce mexe. Olha o seu numero na tabela abaixo:

| Sessao | Founder | RESUME_DATASET | OUTPUT_DATASET_SLUG | Status |
|---|---|---|---|---|
| 1 | pedroafonso2 | `None` | `caracal-base-3b-s01` | done |
| 2 | devknz | `"pedroafonso2/caracal-base-3b-s01"` | `caracal-base-3b-s02` | done |
| 3 | arturpn | `"devknz/caracal-base-3b-s02"` | `arturpn` (slug nao-padrao) | done |
| 4 | vitorscrt | `"arturpn/arturpn"` | `caracal-base-3b-s04` | pending |
| 5 | aletlucas | `"vitorscrt/caracal-base-3b-s04"` | `caracal-base-3b-v0` | pending |

**Exemplo Vitor (sessao 4, proxima):**

```python
SESSION = 4
FOUNDER_HANDLE = "vitorscrt"
RESUME_DATASET = "arturpn/arturpn"  # slug nao-padrao: Arthur subiu como arturpn/arturpn em vez de arturpn/caracal-base-3b-s03
OUTPUT_DATASET_SLUG = "caracal-base-3b-s04"
STEPS = 900
```

Nao mexe em mais nada do notebook. As outras cells ja sabem o que fazer.

---

## 4. Save & Run All (1 click)

Clica botao **"Save Version"** no canto superior direito -> escolhe **"Save & Run All (Commit)"**.

Notebook comeca a rodar. Pode fechar a aba do navegador - continua rodando no servidor Kaggle.

---

## 5. Acompanhar (opcional)

Volta no notebook quando quiser. Aba "Output" mostra logs ao vivo.

Voce vai ver:
- ~3 min instalando deps
- ~30s clonando repo
- ~30s baixando Qwen base do HF
- ~2 min preparando datasets (PrimeVul + BigVul + DiverseVul, ja com decontam)
- ~2 min tokenizando + packing
- A partir dai: `step 1/900, step 2/900, ..., step 900/900` (cada step ~40 segundos)
- Tempo total: **~10 horas**

Checkpoint salvo a cada 50 steps em `/kaggle/working/ckpt-out`. Se kernel cair antes do fim, ultimo ckpt salvo serve.

---

## 6. Apos terminar (~10h depois)

O notebook ao fim publica seu Kaggle Dataset automaticamente:

```
pedroafonso2/caracal-bench-s0X    <- atualizado pelo TPU bench (se rodando)
SEU_HANDLE/caracal-base-3b-sNN    <- voce publicou
```

### 6a. Confirmar dataset publicado

Vai em `https://www.kaggle.com/datasets/SEU_HANDLE/caracal-base-3b-sNN`. Tem que existir, ser publico, e ter arquivos de adapter LoRA dentro (`adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`, etc).

### 6b. Avisar grupo + PR

1. `git checkout dev && git pull`
2. `git checkout -b book-slot-N-done`
3. Editar [SCHEDULE.md](SCHEDULE.md): trocar `in-progress · SEU_HANDLE · TIMESTAMP` por `done · SEU_HANDLE/caracal-base-3b-sNN · TIMESTAMP`
4. `git push -u origin book-slot-N-done`
5. PR contra `dev` · self-merge
6. Avisa proximo founder no grupo: "s0X terminou, slug = SEU_HANDLE/caracal-base-3b-sNN, tudo seu"

### 6c. Verificar eval shadow (opcional)

Se shadow TPU estiver rodando:

```bash
kaggle datasets download -d pedroafonso2/caracal-shadow-eval -p . --unzip
cat session_0X.json
```

Voce ve `mean_ppl`, `cwe_hit_rate`, `pass_at_1` da sua sessao. Compara com a anterior pra ver se o modelo melhorou.

---

## Quando algo da errado

| Situacao | Acao |
|---|---|
| RESUME_DATASET nao baixa (`403 Forbidden`) | Anterior nao publicou ainda OU publicou como private. Avisa anterior. |
| Treino crashou em step X | Ultimo ckpt salvo em `/kaggle/working/ckpt-out/checkpoint-X`. Publica esse mesmo com `kaggle datasets create -p /kaggle/working/ckpt-out/checkpoint-X --public`. Proximo retoma de la. |
| Kernel timeout 12h | Mesma coisa: pega ckpt mais recente, publica, proximo retoma. |
| `CUDA out of memory` | Edita o cmd do continued_pretrain.py pra `--batch-size 1 --grad-accum 32` (metade do batch). |
| Nao sei se posso comecar | Confirma com Pedro/grupo. SCHEDULE.md e fonte da verdade. |

---

## Resumo super resumido

```
1. confere kaggle datasets list -s "caracal-base-3b"     (anterior publicou?)
2. abre kaggle.com/code, new notebook, importa o notebook do repo
3. setta GPU T4 x2 + Internet + Persistence
4. edita 5 vars no topo (olha tabela acima)
5. Save & Run All
6. ~10h depois, dataset publicado
7. PR SCHEDULE.md done + avisa proximo founder
```
