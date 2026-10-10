#!/usr/bin/env python3
"""
Generate simulated A/B experiment data for a CTR uplift scenario.

Scenario (fictional, for portfolio / method demo only):
  E-commerce app tests a new recommender (treatment) vs the old one (control).
  Metric: user-level click-through rate (CTR).
  CUPED covariate: pre-experiment week CTR (pre_ctr).

This is NOT real company data. Ground-truth treatment effect is known by construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(
    n_per_arm: int = 5000,
    true_effect: float = 0.012,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_per_arm

    # Latent user quality drives both pre and post CTR (creates pre/post correlation).
    user_quality = rng.beta(2, 5, n * 2)

    pre_ctr_control = user_quality[:n] * 0.15 + rng.normal(0.05, 0.02, n)
    pre_ctr_treat = user_quality[n:] * 0.15 + rng.normal(0.05, 0.02, n)
    pre_ctr_control = np.clip(pre_ctr_control, 0, 1)
    pre_ctr_treat = np.clip(pre_ctr_treat, 0, 1)

    post_ctr_control = user_quality[:n] * 0.15 + 0.08 + rng.normal(0, 0.04, n)
    post_ctr_treat = user_quality[n:] * 0.15 + 0.08 + true_effect + rng.normal(0, 0.04, n)
    post_ctr_control = np.clip(post_ctr_control, 0, 1)
    post_ctr_treat = np.clip(post_ctr_treat, 0, 1)

    df_control = pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(n)],
            "group": "control",
            "pre_ctr": pre_ctr_control,
            "post_ctr": post_ctr_control,
        }
    )
    df_treat = pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(n, n * 2)],
            "group": "treatment",
            "pre_ctr": pre_ctr_treat,
            "post_ctr": post_ctr_treat,
        }
    )
    return pd.concat([df_control, df_treat], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated A/B CTR experiment CSV")
    parser.add_argument("--n", type=int, default=5000, help="Users per arm")
    parser.add_argument("--effect", type=float, default=0.012, help="True absolute CTR lift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/ab_experiment.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n_per_arm=args.n, true_effect=args.effect, seed=args.seed)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    ctrl = df[df["group"] == "control"]["post_ctr"]
    treat = df[df["group"] == "treatment"]["post_ctr"]
    print(f"数据生成完成（模拟）: {len(df)} 条 → {out}")
    print(f"对照组 post_ctr 均值: {ctrl.mean():.4f}")
    print(f"实验组 post_ctr 均值: {treat.mean():.4f}")
    print(f"构造真实提升: {args.effect:.4f}")
    print("说明: 本文件为合成数据，仅用于统计方法演示。")


if __name__ == "__main__":
    main()
