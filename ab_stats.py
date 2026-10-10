"""
Statistical helpers for A/B analysis: AA, SRM, t-test, CUPED, bootstrap CI, power, sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


ALPHA_DEFAULT = 0.05
POWER_DEFAULT = 0.80
N_BOOTSTRAP_DEFAULT = 5000
RNG_SEED = 42


@dataclass
class AAResult:
    control_mean: float
    treat_mean: float
    t_stat: float
    p_value: float
    passed: bool


@dataclass
class SRMResult:
    n_control: int
    n_treat: int
    expected_ratio: float
    chi2: float
    p_value: float
    passed: bool


@dataclass
class TTestResult:
    control_mean: float
    treat_mean: float
    control_std: float
    treat_std: float
    diff: float
    relative_lift_pct: float
    se: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    cohen_d: float
    control_var: float
    treat_var: float


@dataclass
class CUPEDResult:
    theta: float
    pre_post_corr: float
    control_cuped: np.ndarray
    treat_cuped: np.ndarray
    diff: float
    se: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    var_reduction_pct: float
    ci_width_reduction_pct: float
    control_var_raw: float
    control_var_cuped: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_diff_mean: float


def load_experiment(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    control = df[df["group"] == "control"].copy()
    treat = df[df["group"] == "treatment"].copy()
    return df, control, treat


def run_aa_test(
    control_pre: np.ndarray,
    treat_pre: np.ndarray,
    alpha: float = ALPHA_DEFAULT,
) -> AAResult:
    t_stat, p_value = stats.ttest_ind(control_pre, treat_pre, equal_var=False)
    return AAResult(
        control_mean=float(np.mean(control_pre)),
        treat_mean=float(np.mean(treat_pre)),
        t_stat=float(t_stat),
        p_value=float(p_value),
        passed=bool(p_value > alpha),
    )


def run_srm_check(
    n_control: int,
    n_treat: int,
    expected_ratio: float = 0.5,
    alpha: float = ALPHA_DEFAULT,
) -> SRMResult:
    """
    Sample Ratio Mismatch (SRM) check via chi-square goodness of fit.
    expected_ratio is the intended share of the control group (default 50/50).
    """
    n_total = n_control + n_treat
    expected = np.array([expected_ratio * n_total, (1.0 - expected_ratio) * n_total])
    observed = np.array([n_control, n_treat], dtype=float)
    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    return SRMResult(
        n_control=n_control,
        n_treat=n_treat,
        expected_ratio=expected_ratio,
        chi2=float(chi2),
        p_value=float(p_value),
        passed=bool(p_value > alpha),
    )


def run_ttest(
    control_post: np.ndarray,
    treat_post: np.ndarray,
    alpha: float = ALPHA_DEFAULT,
) -> TTestResult:
    t_stat, p_value = stats.ttest_ind(treat_post, control_post, equal_var=False)
    diff = float(np.mean(treat_post) - np.mean(control_post))
    ctrl_mean = float(np.mean(control_post))
    relative_lift_pct = diff / ctrl_mean * 100.0 if ctrl_mean != 0 else float("nan")

    se = float(
        np.sqrt(
            np.var(control_post, ddof=1) / len(control_post)
            + np.var(treat_post, ddof=1) / len(treat_post)
        )
    )
    z = stats.norm.ppf(1 - alpha / 2)
    pooled_std = float(
        np.sqrt((np.var(control_post, ddof=1) + np.var(treat_post, ddof=1)) / 2)
    )
    cohen_d = diff / pooled_std if pooled_std > 0 else float("nan")

    return TTestResult(
        control_mean=ctrl_mean,
        treat_mean=float(np.mean(treat_post)),
        control_std=float(np.std(control_post, ddof=1)),
        treat_std=float(np.std(treat_post, ddof=1)),
        diff=diff,
        relative_lift_pct=relative_lift_pct,
        se=se,
        ci_low=diff - z * se,
        ci_high=diff + z * se,
        t_stat=float(t_stat),
        p_value=float(p_value),
        cohen_d=cohen_d,
        control_var=float(np.var(control_post, ddof=1)),
        treat_var=float(np.var(treat_post, ddof=1)),
    )


def _cuped_adjust(
    post: np.ndarray,
    pre: np.ndarray,
    theta: float,
    pre_mean: float,
) -> np.ndarray:
    return post - theta * (pre - pre_mean)


def bootstrap_cuped_diff(
    control_post: np.ndarray,
    control_pre: np.ndarray,
    treat_post: np.ndarray,
    treat_pre: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for the CUPED uplift.
    Theta and global pre-mean are re-estimated inside each bootstrap replicate
    (pooled over resampled control + treatment), which better reflects estimation uncertainty.
    """
    rng = np.random.default_rng(seed)
    n_c, n_t = len(control_post), len(treat_post)
    diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx_c = rng.integers(0, n_c, size=n_c)
        idx_t = rng.integers(0, n_t, size=n_t)
        c_post, c_pre = control_post[idx_c], control_pre[idx_c]
        t_post, t_pre = treat_post[idx_t], treat_pre[idx_t]

        pre_all = np.concatenate([c_pre, t_pre])
        post_all = np.concatenate([c_post, t_post])
        var_pre = np.var(pre_all, ddof=0)
        if var_pre <= 0:
            theta = 0.0
        else:
            theta = float(np.cov(pre_all, post_all, ddof=0)[0, 1] / var_pre)
        pre_mean = float(np.mean(pre_all))

        c_adj = _cuped_adjust(c_post, c_pre, theta, pre_mean)
        t_adj = _cuped_adjust(t_post, t_pre, theta, pre_mean)
        diffs[i] = float(np.mean(t_adj) - np.mean(c_adj))

    low = float(np.quantile(diffs, alpha / 2))
    high = float(np.quantile(diffs, 1 - alpha / 2))
    return low, high, float(np.mean(diffs))


def run_cuped(
    control_post: np.ndarray,
    control_pre: np.ndarray,
    treat_post: np.ndarray,
    treat_pre: np.ndarray,
    alpha: float = ALPHA_DEFAULT,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = RNG_SEED,
) -> CUPEDResult:
    pre_all = np.concatenate([control_pre, treat_pre])
    post_all = np.concatenate([control_post, treat_post])
    var_pre = float(np.var(pre_all, ddof=0))
    theta = float(np.cov(pre_all, post_all, ddof=0)[0, 1] / var_pre) if var_pre > 0 else 0.0
    pre_post_corr = float(np.corrcoef(pre_all, post_all)[0, 1])
    pre_mean = float(np.mean(pre_all))

    ctrl_cuped = _cuped_adjust(control_post, control_pre, theta, pre_mean)
    treat_cuped = _cuped_adjust(treat_post, treat_pre, theta, pre_mean)

    t_stat, p_value = stats.ttest_ind(treat_cuped, ctrl_cuped, equal_var=False)
    diff = float(np.mean(treat_cuped) - np.mean(ctrl_cuped))
    se = float(
        np.sqrt(
            np.var(ctrl_cuped, ddof=1) / len(ctrl_cuped)
            + np.var(treat_cuped, ddof=1) / len(treat_cuped)
        )
    )
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = diff - z * se, diff + z * se

    control_var_raw = float(np.var(control_post, ddof=1))
    control_var_cuped = float(np.var(ctrl_cuped, ddof=1))
    var_reduction_pct = (
        (1.0 - control_var_cuped / control_var_raw) * 100.0 if control_var_raw > 0 else 0.0
    )

    raw_width = 2 * z * float(
        np.sqrt(
            np.var(control_post, ddof=1) / len(control_post)
            + np.var(treat_post, ddof=1) / len(treat_post)
        )
    )
    cuped_width = ci_high - ci_low
    ci_width_reduction_pct = (1.0 - cuped_width / raw_width) * 100.0 if raw_width > 0 else 0.0

    boot_low, boot_high, boot_mean = bootstrap_cuped_diff(
        control_post,
        control_pre,
        treat_post,
        treat_pre,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )

    return CUPEDResult(
        theta=theta,
        pre_post_corr=pre_post_corr,
        control_cuped=ctrl_cuped,
        treat_cuped=treat_cuped,
        diff=diff,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        t_stat=float(t_stat),
        p_value=float(p_value),
        var_reduction_pct=var_reduction_pct,
        ci_width_reduction_pct=ci_width_reduction_pct,
        control_var_raw=control_var_raw,
        control_var_cuped=control_var_cuped,
        bootstrap_ci_low=boot_low,
        bootstrap_ci_high=boot_high,
        bootstrap_diff_mean=boot_mean,
    )


def calc_sample_size_per_arm(
    baseline: float,
    mde: float,
    alpha: float = ALPHA_DEFAULT,
    power: float = POWER_DEFAULT,
) -> int:
    """Two-proportion sample size per arm (equal allocation), absolute MDE on rate scale."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1, p2 = baseline, baseline + mde
    pooled = (p1 + p2) / 2.0
    n = (z_alpha + z_beta) ** 2 * 2 * pooled * (1.0 - pooled) / (mde**2)
    return int(np.ceil(n))


def power_table(
    baseline: float,
    mdes: Iterable[float],
    daily_active_per_arm: int = 25000,
    alpha: float = ALPHA_DEFAULT,
    power: float = POWER_DEFAULT,
) -> pd.DataFrame:
    rows = []
    for mde in mdes:
        n = calc_sample_size_per_arm(baseline, mde, alpha=alpha, power=power)
        days = int(np.ceil(n / daily_active_per_arm))
        rows.append(
            {
                "mde": mde,
                "mde_pct": mde * 100,
                "n_per_arm": n,
                "days_at_5e4_dau": days,
            }
        )
    return pd.DataFrame(rows)


def cuped_variance_reduction_theory(rho: float) -> float:
    """Theoretical relative variance reduction of CUPED: rho^2 (for optimally chosen theta)."""
    return float(rho**2)


def correlation_sensitivity_table(
    rhos: Iterable[float] | None = None,
) -> pd.DataFrame:
    """
    How much variance CUPED can remove as a function of pre/post correlation.
    Useful for deciding whether collecting a covariate is worth the engineering cost.
    """
    if rhos is None:
        rhos = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for rho in rhos:
        reduction = cuped_variance_reduction_theory(rho)
        # Effective sample size multiplier ≈ 1 / (1 - rho^2)
        ess_mult = 1.0 / (1.0 - reduction) if reduction < 1.0 else float("inf")
        rows.append(
            {
                "rho": rho,
                "var_reduction_pct": reduction * 100.0,
                "ess_multiplier": ess_mult,
            }
        )
    return pd.DataFrame(rows)


def effect_size_label(cohen_d: float) -> str:
    ad = abs(cohen_d)
    if ad < 0.2:
        return "小"
    if ad < 0.5:
        return "中"
    return "大"
