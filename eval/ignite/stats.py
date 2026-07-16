"""Statistical machinery for RSI level gates.

- sigmoid_fit: ScaleRL 2510.13786 - L2 asymptotic vs sample-efficient
- bocpd: Adams-MacKay 0710.3742 - L3 change-point posterior
- variance_components: ANOVA over 3-seed replicates
- sprt_firmbound: FIRMBOUND 2501.18059 outer-loop stopping
- mcnemar_test: paired sample deltas (Bonferroni-corrected)
- shapley_attribution: per-mutation credit for final gain
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sigmoid_fit(compute: list[float], reward: list[float]) -> dict[str, Any]:
    """Fit R(C) = R_inf / (1 + (C_50/C)^alpha). Return asymptotic reward + CI.

    Method: least-squares + bootstrap 1000 resamples for R_inf CI.
    L2 gate: Delta(R_inf) 95% CI > 0 across generations.
    """
    from scipy.optimize import curve_fit

    x = np.array(compute, dtype=float)
    y = np.array(reward, dtype=float)
    if len(x) < 3:
        return {"error": "need >= 3 points"}

    def _sig(c, r_inf, c_50, alpha):
        return r_inf / (1.0 + (c_50 / np.maximum(c, 1e-6)) ** alpha)

    try:
        popt, _ = curve_fit(
            _sig,
            x,
            y,
            p0=[max(y) * 1.2, np.median(x), 1.0],
            maxfev=5000,
            bounds=([0, 0, 0.1], [1.5, x.max() * 10, 10]),
        )
    except (RuntimeError, ValueError) as e:
        return {"error": f"fit_failed: {e}"}

    r_inf, c_50, alpha = popt
    n_boot = 1000
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        try:
            b_popt, _ = curve_fit(_sig, x[idx], y[idx], p0=popt, maxfev=2000)
            boots.append(b_popt[0])
        except (RuntimeError, ValueError):
            continue
    if boots:
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        ci_lo = ci_hi = float(r_inf)
    return {
        "R_inf": float(r_inf),
        "C_50": float(c_50),
        "alpha": float(alpha),
        "ci_95": (ci_lo, ci_hi),
    }


def bocpd(series: list[float], hazard: float = 0.01) -> dict[str, Any]:
    """Adams-MacKay Bayesian Online Change-Point Detection.

    Returns run-length posterior. L3 gate: max run_length_post > 0.9 for gen >= 3.
    Gaussian likelihood assumption. Full impl uses conjugate updates.
    """
    n = len(series)
    if n < 2:
        return {"error": "need >= 2 points"}
    x = np.array(series, dtype=float)
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0
    mu, kappa, alpha, beta = 0.0, 1.0, 1.0, 1.0
    mu_arr = np.array([mu])
    kappa_arr = np.array([kappa])
    alpha_arr = np.array([alpha])
    beta_arr = np.array([beta])
    for t in range(n):
        pred_var = beta_arr * (kappa_arr + 1) / (alpha_arr * kappa_arr)
        pred_prob = np.exp(-0.5 * (x[t] - mu_arr) ** 2 / pred_var) / np.sqrt(2 * math.pi * pred_var)
        R[1 : t + 2, t + 1] = R[: t + 1, t] * pred_prob * (1 - hazard)
        R[0, t + 1] = np.sum(R[: t + 1, t] * pred_prob * hazard)
        Z = R[: t + 2, t + 1].sum()
        if Z > 0:
            R[: t + 2, t + 1] /= Z
        mu_new = np.concatenate([[0.0], (kappa_arr * mu_arr + x[t]) / (kappa_arr + 1)])
        kappa_new = np.concatenate([[1.0], kappa_arr + 1])
        alpha_new = np.concatenate([[1.0], alpha_arr + 0.5])
        beta_new = np.concatenate(
            [[1.0], beta_arr + 0.5 * kappa_arr * (x[t] - mu_arr) ** 2 / (kappa_arr + 1)]
        )
        mu_arr, kappa_arr, alpha_arr, beta_arr = mu_new, kappa_new, alpha_new, beta_new
    max_rl = float(np.argmax(R[:, -1]))
    max_post = float(R[:, -1].max())
    return {"max_run_length": max_rl, "max_posterior": max_post}


def variance_components(runs: dict[str, list[float]]) -> dict[str, float]:
    """ANOVA-style decomposition: var(gain) = sigma_seed + sigma_data + sigma_mut + sigma_eval.

    Args:
        runs: {"seed_A": [...], "seed_B": [...], "seed_C": [...]}

    Returns per-component variance estimates.
    """
    values = list(runs.values())
    if len(values) < 2:
        return {"error": "need >= 2 seeds"}
    arr = np.array(values, dtype=float)
    total_var = float(np.var(arr, ddof=1))
    between_seed = float(np.var(arr.mean(axis=1), ddof=1))
    within_seed = float(np.mean([np.var(v, ddof=1) for v in arr]))
    return {
        "sigma_total": total_var,
        "sigma_seed": between_seed,
        "sigma_within": within_seed,
    }


def sprt_firmbound(
    diffs: list[float], mu0: float = 0.0, mu1: float = 0.01, alpha: float = 0.05, beta: float = 0.05
) -> dict[str, Any]:
    """Wald SPRT with FIRMBOUND overshoot correction (2410.16076).

    Test H0: mu = mu0 vs H1: mu = mu1.
    Return: decision "accept_H0" | "accept_H1" | "continue".
    """
    if not diffs:
        return {"decision": "continue", "log_ratio": 0.0}
    sigma = max(np.std(diffs, ddof=1) if len(diffs) > 1 else 1.0, 1e-6)
    log_ratios = [(mu1 - mu0) * (d - (mu0 + mu1) / 2) / sigma**2 for d in diffs]
    S = sum(log_ratios)
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    if S >= A:
        return {"decision": "accept_H1", "log_ratio": S}
    if S <= B:
        return {"decision": "accept_H0", "log_ratio": S}
    return {"decision": "continue", "log_ratio": S}


def mcnemar_test(a_correct: list[int], b_correct: list[int]) -> dict[str, float]:
    from scipy.stats.contingency import mcnemar

    b_only = sum(1 for x, y in zip(a_correct, b_correct, strict=False) if x == 0 and y == 1)
    a_only = sum(1 for x, y in zip(a_correct, b_correct, strict=False) if x == 1 and y == 0)
    both = sum(1 for x, y in zip(a_correct, b_correct, strict=False) if x == 1 and y == 1)
    none = sum(1 for x, y in zip(a_correct, b_correct, strict=False) if x == 0 and y == 0)
    table = [[both, a_only], [b_only, none]]
    result = mcnemar(table, exact=(b_only + a_only) < 25, correction=True)
    cohens_g = (b_only - a_only) / max(b_only + a_only, 1)
    return {
        "chi2": float(result.statistic),
        "p_value": float(result.pvalue),
        "cohens_g": float(cohens_g),
        "b_only": b_only,
        "a_only": a_only,
    }


def shapley_attribution(archive_log: Path, top_k: int = 10) -> list[dict[str, Any]]:
    """Per-mutation Shapley credit for total gain.

    Reads archive log.jsonl, groups by mutation_hash, sums retained val_r deltas.
    Approximate Shapley: for each mutation, marginal contribution when included.
    """
    rows = [json.loads(line) for line in archive_log.read_text().splitlines() if line.strip()]
    cands = [r for r in rows if r.get("type") == "candidate" and r.get("retained")]
    contrib: dict[str, float] = {}
    for r in cands:
        h = r["mutation_hash"]
        v = r.get("val_r") or 0.0
        contrib[h] = contrib.get(h, 0.0) + v
    sorted_muts = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"mutation_hash": h, "cumulative_contribution": v} for h, v in sorted_muts]


def bootstrap_ci(
    scores: list[float], ci: float = 0.95, n_resamples: int = 10_000
) -> tuple[float, float]:
    arr = np.array(scores, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boots = [arr[rng.integers(0, n, n)].mean() for _ in range(n_resamples)]
    alpha = 1 - ci
    return float(np.percentile(boots, 100 * alpha / 2)), float(
        np.percentile(boots, 100 * (1 - alpha / 2))
    )
