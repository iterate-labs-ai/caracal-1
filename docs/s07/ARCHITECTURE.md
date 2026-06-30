# Caracal-Agent s07 - Architecture

In-Model Agentic Recursive Cyber Specialist.

Base: Qwen2.5-3B + Caracal s05 LoRA (mantém cyber knowledge das 5 sessions).

Phase 2 do Caracal-1, começa apos s05 ship + benchmark Alexandre.

## 6 Layers

```
Qwen2.5-3B + s05 LoRA (FROZEN)
        │
        ▼
L1 Latent Recursion       (LoopUS 2605.11011 + Retrofitted 2511.07384)
        │
        ▼
L2 Reasoning Tokens       (DeepSeek-R1 2501.12948 + Fast Quiet-STaR 2505.17746)
        │
        ▼
L3 In-Model Tool Calling  (Hammer 2410.04587 + xLAM 2409.03215 + LoopTool 2511.09148)
        │
        ▼
L4 RLVR Minerva-style     (Minerva 2602.00513)
        │
        ▼
L5 RSD Recursive Self-Improve (DGM 2505.22954 + Self-Improving Transformers 2502.01612)
```

## Special tokens (extend Qwen tokenizer)

```
<think> </think>           - DeepSeek-R1 reasoning trace
<verify> </verify>          - self-check fit
<halt confidence="0.X"/>   - adaptive recursion stop
<tool_call> </tool_call>    - function call boundary
<observation> </observation> - tool result wrap
\boxed{CWE-NNN}             - Process Reward (2504.16828) final answer
```

## 10 Cyber Tools (function calling registry)

1. `cwe_lookup(id)` - desc + parents + children + view 1003
2. `cwe_tree_path(id)` - ancestors chain
3. `cwe_search_kw(text)` - top-5 candidates por keyword
4. `attack_lookup(id)` - MITRE ATT&CK technique
5. `cve_similar(desc)` - vector search top-K past CVEs
6. `cwe_micro_rubric(id)` - "quando aplica X?" rules
7. `cvss_calc(vector)` - severity calc
8. `verify_fit(cve, cwe)` - separate verifier judges
9. `cwe_view_filter(view)` - top-25 / top-1000 filter
10. `nvd_fetch(cve_id)` - official NVD entry

## Reward function (Minerva RLVR)

Hierarchical CWE partial credit:

```
exact CWE match               = 1.0
parent/ancestor match          = 0.5
sibling (shared parent) match  = 0.3
shared view 1003               = 0.1
format bonus (regex + boxed)   = +0.1
length penalty over 256 tok    = -0.05/32
tool args valid                = +0.05
```

## Stack expected CTI-RCM

| Layer | Source paper | Estimated gain | Cumulative |
|---|---|---|---|
| Caracal s05 baseline | (atual) | - | 42.7% |
| L1 Latent Recursion | LoopUS + Ouro + Retrofitted | +5-8pp | 48-51% |
| L2 Reasoning Tokens | DeepSeek-R1 + Fast Quiet-STaR | +5-10pp | 53-61% |
| L3 Tool Calling | Hammer + xLAM + LoopTool | +8-12pp | 61-73% |
| L4 RLVR Minerva | Minerva proven +20pp on Llama-3.1-8B | +12-18pp | 73-85% |
| L5 RSD + DGM | Self-Improving Transformers + DGM | +5-7pp | **78-92%** |
| Test-time FlashThink | FlashThink early-exit | +1-3pp | **79-95%** |

Realistic landing: **75-85% CTI-RCM** (peer Sec-Gemini 86%, Mythos 83%).

## Compute: Kaggle T4 x2 only

5 founders relay, 7 sessions x 12h = 84h total. No Modal.

See HANDOFF.md per-session breakdown.

## Key decisions (locked)

- Base: Qwen2.5-3B (continuidade s05, nao switch HRM-Text)
- Reward: hierarchical (R5.2 design)
- Tool format: JSON xLAM-style (proven em 3B)
- Recursion: LoopUS post-train conversion (mantem s05 frozen)
- RL: Minerva RLVR (cyber paper irmao)
