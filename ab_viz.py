"""Visualization for A/B + CUPED report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams

from ab_stats import (
    CUPEDResult,
    TTestResult,
    calc_sample_size_per_arm,
    correlation_sensitivity_table,
)


def configure_fonts() -> None:
    """Prefer Noto Sans CJK SC when available (Linux); fall back gracefully."""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


BLUE = "#2563eb"
GREEN = "#16a34a"
GRAY = "#6b7280"
ORANGE = "#ea580c"


def save_report(
    control_post: np.ndarray,
    treat_post: np.ndarray,
    ttest: TTestResult,
    cuped: CUPEDResult,
    sensitivity: pd.DataFrame,
    observed_rho: float,
    baseline_ctr: float,
    alpha: float,
    power: float,
    out_path: str | Path,
) -> Path:
    configure_fonts()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "A/B 实验分析报告 — 推荐算法 CTR（模拟数据）",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 1. CTR distribution
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 0.35, 40)
    ax1.hist(control_post, bins=bins, alpha=0.6, color=GRAY, label="对照组", density=True)
    ax1.hist(treat_post, bins=bins, alpha=0.6, color=BLUE, label="实验组", density=True)
    ax1.axvline(control_post.mean(), color=GRAY, linestyle="--", linewidth=1.5)
    ax1.axvline(treat_post.mean(), color=BLUE, linestyle="--", linewidth=1.5)
    ax1.set_title("CTR 分布对比", fontweight="bold")
    ax1.set_xlabel("CTR")
    ax1.legend(fontsize=9)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # 2. CI comparison: normal approx vs bootstrap CUPED
    ax2 = fig.add_subplot(gs[0, 1])
    methods = ["标准 t-test", "CUPED\n(正态近似)", "CUPED\n(Bootstrap)"]
    diffs = [ttest.diff, cuped.diff, cuped.bootstrap_diff_mean]
    cis = [
        (ttest.ci_low, ttest.ci_high),
        (cuped.ci_low, cuped.ci_high),
        (cuped.bootstrap_ci_low, cuped.bootstrap_ci_high),
    ]
    colors = [GRAY, GREEN, ORANGE]
    for i, (d, ci, color) in enumerate(zip(diffs, cis, colors)):
        ax2.errorbar(
            i,
            d,
            yerr=[[d - ci[0]], [ci[1] - d]],
            fmt="o",
            color=color,
            capsize=7,
            capthick=2,
            markersize=7,
            linewidth=2,
        )
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylabel("CTR 提升量")
    ax2.set_title("效应量置信区间对比", fontweight="bold")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # 3. Variance reduction bar
    ax3 = fig.add_subplot(gs[0, 2])
    categories = ["标准", "CUPED"]
    variances = [cuped.control_var_raw, cuped.control_var_cuped]
    bars = ax3.bar(categories, variances, color=[GRAY, GREEN], alpha=0.85, width=0.5)
    for bar, v in zip(bars, variances):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(variances) * 0.02,
            f"{v:.5f}",
            ha="center",
            fontsize=9,
        )
    ax3.set_title(f"方差缩减\n{cuped.var_reduction_pct:.1f}%", fontweight="bold")
    ax3.set_ylabel("对照组方差")
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    # 4. Sample size vs MDE
    ax4 = fig.add_subplot(gs[1, 0])
    mde_range = np.linspace(0.003, 0.025, 80)
    n_range = [calc_sample_size_per_arm(baseline_ctr, m, alpha=alpha, power=power) for m in mde_range]
    ax4.plot(mde_range * 100, n_range, color=BLUE, linewidth=2.5)
    ax4.fill_between(mde_range * 100, n_range, alpha=0.1, color=BLUE)
    for mde in (0.005, 0.010, 0.015):
        n = calc_sample_size_per_arm(baseline_ctr, mde, alpha=alpha, power=power)
        ax4.scatter(mde * 100, n, s=70, color=BLUE, zorder=5)
        ax4.annotate(
            f"MDE={mde*100:.1f}%\nn={n:,}",
            (mde * 100, n),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )
    ax4.set_xlabel("MDE (%)")
    ax4.set_ylabel("单组样本量")
    ax4.set_title(f"样本量规划 (α={alpha}, power={power})", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax4.spines[s].set_visible(False)

    # 5. CUPED sensitivity to rho
    ax5 = fig.add_subplot(gs[1, 1])
    sens = sensitivity if sensitivity is not None else correlation_sensitivity_table()
    ax5.plot(sens["rho"], sens["var_reduction_pct"], color=GREEN, linewidth=2.5, marker="o")
    ax5.axvline(observed_rho, color=ORANGE, linestyle="--", linewidth=1.5, label=f"本数据 ρ={observed_rho:.2f}")
    ax5.set_xlabel("pre/post 相关系数 ρ")
    ax5.set_ylabel("理论方差缩减 (%)")
    ax5.set_title("CUPED 灵敏度：ρ → 方差缩减", fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    for s in ("top", "right"):
        ax5.spines[s].set_visible(False)

    # 6. Summary card
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary = [
        ("数据说明", "模拟电商 CTR 实验", GRAY),
        ("CTR 绝对提升", f"{ttest.diff:+.4f} ({ttest.relative_lift_pct:+.2f}%)", BLUE),
        ("t-test p", f"{ttest.p_value:.4g}", GREEN if ttest.p_value < alpha else ORANGE),
        ("CUPED p", f"{cuped.p_value:.4g}", GREEN if cuped.p_value < alpha else ORANGE),
        ("Bootstrap CI", f"[{cuped.bootstrap_ci_low:.4f}, {cuped.bootstrap_ci_high:.4f}]", ORANGE),
        ("方差缩减", f"{cuped.var_reduction_pct:.1f}% (ρ={observed_rho:.2f})", GREEN),
        ("Cohen's d", f"{ttest.cohen_d:.3f}", GRAY),
    ]
    y_pos = 0.95
    ax6.text(0.05, y_pos, "实验摘要", fontsize=12, fontweight="bold", transform=ax6.transAxes)
    for label, value, color in summary:
        y_pos -= 0.11
        ax6.text(0.05, y_pos, f"{label}:", fontsize=9, color=GRAY, transform=ax6.transAxes)
        ax6.text(
            0.05,
            y_pos - 0.05,
            value,
            fontsize=9,
            fontweight="bold",
            color=color,
            transform=ax6.transAxes,
        )
        y_pos -= 0.04

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
