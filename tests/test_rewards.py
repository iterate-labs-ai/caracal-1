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
    try:
        from eval.ignite.reward import _cwe_tree

        return _cwe_tree()
    except (OSError, ImportError) as e:
        pytest.skip(f"arvore CWE indisponivel: {e}")


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
    from eval.s07 import bench_runner, hier_reward
    from eval.s07.benches._common import normalize_cwe

    assert hier_reward.normalize_cwe is normalize_cwe
    assert bench_runner.normalize_cwe is normalize_cwe
    # o regex novo aceita "CWE 119" com espaco (antigo rejeitava)
    assert normalize_cwe("weakness: CWE 119") == "CWE-119"


def test_hier_space_format(cwe_tree):
    from eval.s07.hier_reward import hier_cwe_reward

    # "CWE 119" sem boxed, exato. Antes do fix do regex vinha 0.0.
    assert hier_cwe_reward("the weakness is CWE 119", "CWE-119", cwe_tree) == 1.0


def test_cyber_rcm_hierarchical(cwe_tree):
    child = next((c for c in cwe_tree.cwe_map if cwe_tree.get_ancestors(c)), None)
    parent = cwe_tree.get_ancestors(child)[0]
    cf, pf = f"CWE-{child}", f"CWE-{parent}"
    assert cyber_rcm_reward(f"\\boxed{{{cf}}}", cf) == 1.0
    assert cyber_rcm_reward(f"\\boxed{{{pf}}}", cf) == pytest.approx(0.6)
    assert cyber_rcm_reward("sei la", cf) == -1.0
