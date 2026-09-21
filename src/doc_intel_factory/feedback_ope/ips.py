"""IPS / SNIPS off-policy estimators for review policies."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .logging_policy import ReviewDecision, target_policy_propensity


def _reward_for_action(d: ReviewDecision, action: int) -> float:
    """Review yields oracle improvement; auto-accept yields 0 improvement."""
    return float(d.reward) if action == 1 else 0.0


def ips_estimate(
    decisions: Sequence[ReviewDecision],
    *,
    threshold: float = 0.4,
) -> dict[str, float]:
    """Inverse propensity scoring for target review policy value.

    Action rewards: r(1)=improvement, r(0)=0. Importance weight on logged action.
    """
    weights = []
    weighted_rewards = []
    for d in decisions:
        pi_t_1 = target_policy_propensity(d.uncertainty, threshold=threshold)
        pi_t_a = pi_t_1 if d.action == 1 else (1.0 - pi_t_1)
        pi_0_a = d.propensity if d.action == 1 else (1.0 - d.propensity)
        w = pi_t_a / max(pi_0_a, 1e-6)
        r = _reward_for_action(d, d.action)
        weights.append(w)
        weighted_rewards.append(w * r)
    n = len(decisions) or 1
    est = float(np.sum(weighted_rewards) / n)
    ess = float((np.sum(weights) ** 2) / max(np.sum(np.square(weights)), 1e-12)) if weights else 0.0
    return {
        "estimate": est,
        "n": len(decisions),
        "ess": ess,
        "mean_weight": float(np.mean(weights)) if weights else 0.0,
    }


def snips_estimate(
    decisions: Sequence[ReviewDecision],
    *,
    threshold: float = 0.4,
) -> dict[str, float]:
    """Self-normalized IPS."""
    num = 0.0
    den = 0.0
    weights = []
    for d in decisions:
        pi_t_1 = target_policy_propensity(d.uncertainty, threshold=threshold)
        pi_t_a = pi_t_1 if d.action == 1 else (1.0 - pi_t_1)
        pi_0_a = d.propensity if d.action == 1 else (1.0 - d.propensity)
        w = pi_t_a / max(pi_0_a, 1e-6)
        r = _reward_for_action(d, d.action)
        weights.append(w)
        num += w * r
        den += w
    est = float(num / den) if den > 0 else 0.0
    ess = float((np.sum(weights) ** 2) / max(np.sum(np.square(weights)), 1e-12)) if weights else 0.0
    return {"estimate": est, "n": len(decisions), "ess": ess, "sum_weights": float(den)}


def naive_estimate(
    decisions: Sequence[ReviewDecision],
    *,
    threshold: float = 0.4,
) -> dict[str, float]:
    """Naive on-policy mean: average r(a) on rows where logged action == target action.

    Biased under confounding (uncertainty drives both propensity and reward).
    """
    selected = []
    for d in decisions:
        pi_t_1 = target_policy_propensity(d.uncertainty, threshold=threshold)
        target_action = 1 if pi_t_1 >= 0.5 else 0
        if d.action == target_action:
            selected.append(_reward_for_action(d, d.action))
    return {
        "estimate": float(np.mean(selected)) if selected else 0.0,
        "n_selected": len(selected),
        "n": len(decisions),
    }


def oracle_target_value(
    decisions: Sequence[ReviewDecision],
    *,
    threshold: float = 0.4,
) -> float:
    """True value of target policy: E[π_t(1|x) * r(1,x)]."""
    vals = []
    for d in decisions:
        pi_t_1 = target_policy_propensity(d.uncertainty, threshold=threshold)
        vals.append(pi_t_1 * d.reward)
    return float(np.mean(vals)) if vals else 0.0


def ope_report(decisions: Sequence[ReviewDecision], *, threshold: float = 0.4) -> dict[str, Any]:
    ips = ips_estimate(decisions, threshold=threshold)
    snips = snips_estimate(decisions, threshold=threshold)
    naive = naive_estimate(decisions, threshold=threshold)
    oracle = oracle_target_value(decisions, threshold=threshold)
    return {
        "ips": ips,
        "snips": snips,
        "naive": naive,
        "oracle_target_value": oracle,
        "ips_abs_error": abs(ips["estimate"] - oracle),
        "snips_abs_error": abs(snips["estimate"] - oracle),
        "naive_abs_error": abs(naive["estimate"] - oracle),
        "threshold": threshold,
        "review_rate_logged": float(np.mean([d.action for d in decisions])) if decisions else 0.0,
    }
