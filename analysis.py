#!/usr/bin/env python3
"""
A/B experiment analysis CLI (simulated CTR uplift scenario).

Pipeline:
  1. SRM check (sample ratio mismatch)
  2. AA validation on pre-experiment CTR
  3. Standard Welch t-test on post CTR
  4. CUPED variance reduction + bootstrap CI
  5. Power / MDE sample-size planning
  6. CUPED sensitivity table (rho -> variance reduction)
  7. Save report figure under output/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ab_stats import (
    ALPHA_DEFAULT,
    N_BOOTSTRAP_DEFAULT,
    POWER_DEFAULT,
    correlation_sensitivity_table,
    effect_size_label,
    load_experiment,
    power_table,
    run_aa_test,
    run_cuped,
    run_srm_check,
    run_ttest,
)
from ab_viz import save_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A/B + CUPED analysis on experiment CSV")
    p.add_argument(
        "--data",
        default="data/ab_experiment.csv",
        help="Path to experiment CSV (default: data/ab_experiment.csv)",
    )
    p.add_argument(
        "--output",
        default="output/ab_test_report.png",
        help="Path for report PNG (default: output/ab_test_report.png)",
    )
    p.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    p.add_argument("--power", type=float, default=POWER_DEFAULT)
    p.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Data file not found: {data_path}\n"
            "Run: python generate_data.py"
        )

    df, control, treat = load_experiment(str(data_path))
    ctrl_pre = control["pre_ctr"].to_numpy()
    treat_pre = treat["pre_ctr"].to_numpy()
    ctrl_post = control["post_ctr"].to_numpy()
    treat_post = treat["post_ctr"].to_numpy()

    # ── 1. SRM ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("【Step 1】SRM 检查（Sample Ratio Mismatch）")
    print("=" * 60)
    srm = run_srm_check(len(control), len(treat), expected_ratio=0.5, alpha=args.alpha)
    print(f"  对照组 n={srm.n_control}, 实验组 n={srm.n_treat}")
    print(f"  期望比例 1:1, chi2={srm.chi2:.4f}, p={srm.p_value:.4f}")
    if srm.passed:
        print(f"  结论: p={srm.p_value:.3f} > {args.alpha}，无 SRM，分流比例正常")
    else:
        print(f"  结论: p={srm.p_value:.3f} <= {args.alpha}，存在 SRM，结果可信度存疑")

    # ── 2. AA ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【Step 2】AA 验证 — 实验前 CTR 是否无显著差异")
    print("=" * 60)
    aa = run_aa_test(ctrl_pre, treat_pre, alpha=args.alpha)
    print(f"  对照组 pre_ctr 均值: {aa.control_mean:.4f}")
    print(f"  实验组 pre_ctr 均值: {aa.treat_mean:.4f}")
    print(f"  t={aa.t_stat:.3f}, p={aa.p_value:.4f}")
    if aa.passed:
        print(f"  结论: p={aa.p_value:.3f} > {args.alpha}，AA 通过，分流均匀")
    else:
        print(f"  结论: p={aa.p_value:.3f} <= {args.alpha}，AA 未通过，需排查分流/数据")

    # ── 3. t-test ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【Step 3】标准 Welch t-test（post_ctr）")
    print("=" * 60)
    tt = run_ttest(ctrl_post, treat_post, alpha=args.alpha)
    print(f"  对照组 CTR: {tt.control_mean:.4f}  (std={tt.control_std:.4f})")
    print(f"  实验组 CTR: {tt.treat_mean:.4f}  (std={tt.treat_std:.4f})")
    print(f"  绝对提升:   {tt.diff:+.4f}")
    print(f"  相对提升:   {tt.relative_lift_pct:+.2f}%")
    print(f"  95% CI:     [{tt.ci_low:.4f}, {tt.ci_high:.4f}]")
    print(f"  t={tt.t_stat:.3f}, p={tt.p_value:.4g}")
    print(f"  Cohen's d={tt.cohen_d:.4f}（{effect_size_label(tt.cohen_d)}效应量）")
    if tt.p_value < args.alpha:
        print(f"  结论: 统计显著 (p < {args.alpha})")
    else:
        print(f"  结论: 不显著 (p >= {args.alpha})")

    # ── 4. CUPED + bootstrap ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【Step 4】CUPED 方差缩减 + Bootstrap CI")
    print("=" * 60)
    cuped = run_cuped(
        ctrl_post,
        ctrl_pre,
        treat_post,
        treat_pre,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(f"  theta (协变量系数): {cuped.theta:.4f}")
    print(f"  pre/post 相关系数 ρ: {cuped.pre_post_corr:.4f}")
    print(f"  方差缩减比例: {cuped.var_reduction_pct:.1f}%")
    print(f"  CUPED 绝对提升: {cuped.diff:+.4f}")
    print(f"  CUPED 正态近似 95% CI: [{cuped.ci_low:.4f}, {cuped.ci_high:.4f}]")
    print(
        f"  CUPED Bootstrap 95% CI: [{cuped.bootstrap_ci_low:.4f}, "
        f"{cuped.bootstrap_ci_high:.4f}]  (B={args.bootstrap})"
    )
    print(f"  t={cuped.t_stat:.3f}, p={cuped.p_value:.4g}")
    print(f"  CI 宽度缩减 (相对标准 t-test): {cuped.ci_width_reduction_pct:.1f}%")

    # ── 5. Power / MDE ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【Step 5】样本量规划 (Power Analysis)")
    print("=" * 60)
    mdes = [0.005, 0.008, 0.010, 0.015, 0.020]
    pt = power_table(tt.control_mean, mdes, alpha=args.alpha, power=args.power)
    print(f"  基准 CTR: {tt.control_mean:.4f}, α={args.alpha}, power={args.power}")
    print(f"  {'MDE':>8}  {'单组样本量':>10}  {'实验天数(日活5万,每臂2.5万)':>26}")
    for _, row in pt.iterrows():
        print(f"  {row['mde']:>8.3f}  {int(row['n_per_arm']):>10,}  {int(row['days_at_5e4_dau']):>26d} 天")

    # ── 6. Sensitivity ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("【Step 6】CUPED 灵敏度表（理论：方差缩减 ≈ ρ²）")
    print("=" * 60)
    sens = correlation_sensitivity_table()
    print(f"  {'ρ':>6}  {'理论方差缩减%':>14}  {'等效样本量倍数':>14}")
    for _, row in sens.iterrows():
        print(
            f"  {row['rho']:>6.2f}  {row['var_reduction_pct']:>14.1f}  "
            f"{row['ess_multiplier']:>14.2f}"
        )
    print(
        f"  本数据观测 ρ={cuped.pre_post_corr:.3f} → "
        f"理论缩减 ≈ {cuped.pre_post_corr**2 * 100:.1f}%，"
        f"实测缩减 {cuped.var_reduction_pct:.1f}%"
    )

    # ── 7. Plot ──────────────────────────────────────────────────────────────
    out = save_report(
        ctrl_post,
        treat_post,
        tt,
        cuped,
        sens,
        observed_rho=cuped.pre_post_corr,
        baseline_ctr=tt.control_mean,
        alpha=args.alpha,
        power=args.power,
        out_path=args.output,
    )
    print(f"\n[完成] 图表已保存到 {out}")
    print(f"[说明] 数据为模拟生成（seed 固定），仅用于方法演示，非真实业务数据。")


if __name__ == "__main__":
    main()
