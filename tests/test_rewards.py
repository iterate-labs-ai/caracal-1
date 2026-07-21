"""Regressao dos rewards RLVR (ignite reward.py + s07 hier_reward.py).

Trava o bug do credito parcial hierarquico: normalize_cwe devolvia "CWE-1004"
mas o CWEParser keia por ID cru "1004", entao is_ancestor/shared_parent nunca
casavam e ancestral/irmao caiam pra 0.0 sem crash (numero errado silencioso).
"""

import pytest

from eval.ignite.reward import build_reward_fn, cyber_rcm_reward, math_reward


def test_cyber_rcm_binary():
    assert cyber_rcm_reward("\\boxed{CWE-79}", "CWE-79", hierarchical=False) == 1.0
    assert cyber_rcm_reward("\\boxed{CWE-120}", "CWE-79", hierarchical=False) == 0.0
    # "CWE 79" com espaco tem que normalizar igual (regressao do CWE_RE)
    assert cyber_rcm_reward("final: CWE 79", "CWE-79", hierarchical=False) == 1.0
    # sem CWE nenhum = malformado
    assert cyber_rcm_reward("nao sei", "CWE-79", hierarchical=False) == -1.0


def test_math_reward_smoke():
    assert math_reward("\\boxed{42}", "42") > 0.9
    assert math_reward("\\boxed{99}", "42") == pytest.approx(0.1)


def test_build_reward_fn_cyber_batch():
    fn = build_reward_fn("cyber_rcm")
    # factory usa hierarchical=True: exato=1.0, nao-exato < 1.0 (parcial ou 0)
    out = fn(["\\boxed{CWE-79}", "\\boxed{CWE-120}"], gold=["CWE-79", "CWE-79"])
    assert out[0] == 1.0
    assert out[1] < 1.0


def test_build_reward_fn_unknown():
    with pytest.raises(ValueError):
        build_reward_fn("nope")


# --- credito parcial hierarquico: precisa da arvore CWE (baixa da MITRE) ---


@pytest.fixture(scope="module")
def cwe_tree():
    from eval.s07.cwe_tree_parser import get_cwe_tree

    tree = get_cwe_tree()
    if tree is None:
        pytest.skip("arvore CWE indisponivel (sem XML local nem rede)")
    return tree


def test_hier_partial_credit(cwe_tree):
    from eval.s07.hier_reward import hier_cwe_reward

    child = next((c for c in cwe_tree.cwe_map if cwe_tree.get_ancestors(c)), None)
    assert child, "arvore sem nenhum no com ancestral"
    parent = cwe_tree.get_ancestors(child)[0]
    cf, pf = f"CWE-{child}", f"CWE-{parent}"

    assert hier_cwe_reward(f"\\boxed{{{cf}}}", cf, cwe_tree) == 1.0  # exato
    # ancestral = 0.5 base + 0.1 format bonus. Antes do fix vinha 0.1 (so bonus).
    assert hier_cwe_reward(f"\\boxed{{{pf}}}", cf, cwe_tree) == pytest.approx(0.6)


def test_normalize_cwe_single_source():
    # as 3 copias foram consolidadas em _common: mesma funcao, mesmo id.
    from eval.s07 import hier_reward
    from eval.s07.benches._common import normalize_cwe

    # a 3a copia vivia em bench_runner.py, agora arquivado (era 5 funcoes
    # duplicadas de cti_bench/_common/stats, nenhum notebook o usava)
    assert hier_reward.normalize_cwe is normalize_cwe
    # o regex novo aceita "CWE 119" com espaco (antigo rejeitava)
    assert normalize_cwe("weakness: CWE 119") == "CWE-119"


def test_hier_space_format(cwe_tree):
    from eval.s07.hier_reward import hier_cwe_reward

    # "CWE 119" sem boxed, exato. Antes do fix do regex vinha 0.0.
    assert hier_cwe_reward("the weakness is CWE 119", "CWE-119", cwe_tree) == 1.0


def test_parser_normaliza_na_borda(cwe_tree):
    """A arvore keia por ID cru; antes cada consumidor tirava o prefixo sozinho
    (5 call sites) e quem esquecesse tinha miss silencioso."""
    child = next(c for c in cwe_tree.cwe_map if cwe_tree.get_ancestors(c))
    parent = cwe_tree.get_ancestors(child)[0]
    assert cwe_tree.is_ancestor(f"CWE-{parent}", f"CWE-{child}")
    assert cwe_tree.is_ancestor(parent, child)
    assert cwe_tree.is_ancestor(f"cwe-{parent}", f"CWE-{child}")
    # DOM de 18MB nao fica residente depois do cache
    assert cwe_tree.tree is None and cwe_tree.root is None


def test_mutation_system_prompt_matches_bench():
    """Trava o bug do run 2026-07-20: bench cyber caiu no fallback e treinou
    CVE->CWE com 'You are an AI math tutor'."""
    from train.ignite.mutations import default_mutation

    cyber = default_mutation("cyber_rcm").system_prompt.lower()
    assert "cwe" in cyber and "security" in cyber
    assert "math" not in cyber, "bench cyber caiu no prompt de matematica"
    assert "math" in default_mutation("math").system_prompt.lower()
    # bench desconhecido tem que gritar, nao cair calado no math
    with pytest.raises(ValueError):
        default_mutation("bench_inexistente")


def test_cyber_gen_budget_menor_que_default():
    """Custo do step no GRPO e dominado por completion_len. O cyber responde um
    CWE (~6 tokens), entao nao paga o budget generico."""
    from train.ignite.inner_grpo import BENCH_GEN_BUDGET, DEFAULT_GEN_BUDGET

    cy_prompt, cy_completion = BENCH_GEN_BUDGET["cyber_rcm"]
    _, default_completion = DEFAULT_GEN_BUDGET
    assert default_completion / cy_completion >= 3.0
    assert cy_prompt >= 384  # p90 medido = 169 tokens, folga real


def test_bench_e_treino_usam_o_mesmo_teto_de_geracao():
    """O bench nao importa de train (nao inverter camada), entao o valor e
    espelhado - este teste e o que impede os dois de divergirem."""
    from eval.ignite.benches.cyber_rcm import MAX_NEW
    from train.ignite.inner_grpo import BENCH_GEN_BUDGET

    assert MAX_NEW == BENCH_GEN_BUDGET["cyber_rcm"][1]


def test_cyber_prompt_pede_resposta_curta():
    """Com completion truncado, prompt que convida a divagar corta antes do CWE
    -> reward -1.0 e o RL aprende lixo."""
    from train.ignite.mutations import DEFAULT_SYSTEM_CYBER

    p = DEFAULT_SYSTEM_CYBER.lower()
    assert "concise" in p or "two sentences" in p
    assert "last line" in p


def test_bench_choices_derivam_do_registry():
    """Listas hardcoded de bench ja divergiram uma vez (cyber_rcm entrou no
    C_rsi_outer e nao no inner_grpo)."""
    import inspect

    from eval.ignite.reward import PAIRWISE_REWARDS
    from train.ignite import C_rsi_outer, inner_grpo

    for mod in (inner_grpo, C_rsi_outer):
        src = inspect.getsource(mod)
        assert "sorted(PAIRWISE_REWARDS)" in src, f"{mod.__name__} com choices hardcoded"
    assert "cyber_rcm" in PAIRWISE_REWARDS


def test_cyber_rcm_hierarchical(cwe_tree):
    child = next((c for c in cwe_tree.cwe_map if cwe_tree.get_ancestors(c)), None)
    parent = cwe_tree.get_ancestors(child)[0]
    cf, pf = f"CWE-{child}", f"CWE-{parent}"
    assert cyber_rcm_reward(f"\\boxed{{{cf}}}", cf) == 1.0
    assert cyber_rcm_reward(f"\\boxed{{{pf}}}", cf) == pytest.approx(0.6)
    assert cyber_rcm_reward("sei la", cf) == -1.0


def test_bench_run_schema_canonico():
    """curve.json (GRPO) e trajectory.json (RSI) tem que usar o MESMO schema.
    A drift hier vs hier_score nasceu num PR; o schema unico impede voltar."""
    import inspect

    from eval.ignite.benches.run import CANONICAL_KEYS
    from train.ignite import B_grpo_straight, C_rsi_outer

    assert CANONICAL_KEYS == ("accuracy", "hier_score", "unparsed_frac", "n")
    for mod in (B_grpo_straight, C_rsi_outer):
        src = inspect.getsource(mod)
        assert "['hier']" not in src and "['unparsed']" not in src, f"{mod.__name__} chave nao-canonica"
