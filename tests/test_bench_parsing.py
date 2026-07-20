"""Regressao do parsing dos benches s07 - a classe de bug que da numero errado
sem crashar (n=0, accuracy 0.0%) e passa despercebida num run de horas.

Alimenta o schema REAL de cada dataset + uma resposta correta canned. Se alguem
trocar as chaves de volta (option_a vs answers, answer vs answers), n cai pra 0
ou a accuracy pra 0.0 e o teste pega. Nao baixa dataset nem carrega modelo.
"""

import pytest


class FakeTok:
    """Tokenizer minimo: mcq_chat_prompt so precisa de apply_chat_template."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in msgs)


@pytest.fixture
def tok():
    return FakeTok()


# --- secbench: schema real e answers(list) + label(letra), NAO option_a ---


def test_secbench_schema_scores(monkeypatch, tok):
    from eval.s07.benches import secbench

    rows = [
        {
            "question": "What does CIA stand for?",
            "answers": ["Confidentiality Integrity Availability", "x", "y", "z"],
            "label": "A",
            "language": "en",
        },
        {
            "question": "Which port is HTTPS?",
            "answers": ["21", "443", "23", "25"],
            "label": "B",
            "language": "en",
        },
    ]
    monkeypatch.setattr(secbench, "_load", lambda subset, n: rows)
    # modelo "responde" sempre o gold do row corrente
    seq = iter(["\\boxed{A}", "\\boxed{B}"])
    monkeypatch.setattr(secbench, "generate", lambda *a, **k: next(seq))

    res = secbench.eval_secbench(model=None, tok=tok, n_mcq=10)
    assert res["n"] == 2, f"schema nao casou, n={res['n']} skipped={res['skipped']}"
    assert res["skipped"] == 0
    assert res["accuracy"] == 1.0


def test_secbench_skips_malformed_instead_of_scoring_zero(monkeypatch, tok):
    from eval.s07.benches import secbench

    # row sem gold: tem que entrar em `skipped`, nao virar erro contado como acerto
    monkeypatch.setattr(
        secbench, "_load", lambda subset, n: [{"question": "q", "answers": ["a", "b"]}]
    )
    monkeypatch.setattr(secbench, "generate", lambda *a, **k: "\\boxed{A}")
    res = secbench.eval_secbench(model=None, tok=tok, n_mcq=10)
    assert res["n"] == 0
    assert res["skipped"] == 1


# --- cti_bench RCM: CVE->CWE, gold na coluna GT ---


def test_cti_bench_rcm_scores(monkeypatch, tok):
    from eval.s07.benches import cti_bench

    rows = [
        {"Description": "buffer overflow in foo", "GT": "CWE-120", "Prompt": "classify"},
        {"Description": "xss in bar", "GT": "CWE-79", "Prompt": "classify"},
    ]
    monkeypatch.setattr(cti_bench, "_load_tsv", lambda subset: rows)
    seq = iter(["answer: CWE-120", "the weakness is CWE 79"])  # 2o com espaco
    monkeypatch.setattr(cti_bench, "generate", lambda *a, **k: next(seq))

    res = cti_bench._eval_rcm(model=None, tok=tok, n=10)
    assert res["n"] == 2
    # "CWE 79" com espaco tem que contar como acerto (regressao do CWE_RE)
    assert res["accuracy"] == 1.0


# --- cybersoceval: gold e `answers` LISTA (multi-resposta) ---


class _FakeDS:
    def __init__(self, rows):
        self._rows = rows

    def select(self, _rng):
        return self._rows


def test_cybersoceval_multi_answer_gold(monkeypatch, tok):
    import datasets

    from eval.s07.benches import cybersoceval

    rows = [
        {
            "question": "Which are malware families?",
            "options": ["A. Emotet", "B. Nginx", "C. TrickBot", "D. Bash"],
            "answers": ["A", "C"],  # multi-resposta: o bug lia `answer` e dava ""
        }
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _FakeDS(rows))
    monkeypatch.setattr(cybersoceval, "generate", lambda *a, **k: "\\boxed{A, C}")

    res = cybersoceval.eval_cybersoceval(model=None, tok=tok, n=10)
    scored = [v for v in res.values() if isinstance(v, dict) and "error" not in v]
    assert scored, f"todos os splits falharam: {res}"
    first = scored[0]
    assert first["n"] > 0, "gold multi-resposta nao foi lido (era o 0.0%)"
    assert first["accuracy"] == 1.0
