"""
业务场景：电商 App 测试新版推荐算法（实验组）vs 旧版（对照组）
指标：用户点击率 (CTR)
CUPED 协变量：实验前一周的用户 CTR（pre_ctr）
"""
import pandas as pd
import numpy as np

np.random.seed(42)

N = 5000  # 每组用户数
TRUE_EFFECT = 0.012  # 真实提升效果（1.2个百分点）

# 用户基础特征（影响 pre_ctr 和 post_ctr）
user_quality = np.random.beta(2, 5, N * 2)  # 用户质量分，影响两组

# 实验前 CTR（协变量）——和实验无关，纯反映用户活跃度
pre_ctr_control = user_quality[:N] * 0.15 + np.random.normal(0.05, 0.02, N)
pre_ctr_treat   = user_quality[N:] * 0.15 + np.random.normal(0.05, 0.02, N)
pre_ctr_control = np.clip(pre_ctr_control, 0, 1)
pre_ctr_treat   = np.clip(pre_ctr_treat, 0, 1)

# 实验后 CTR：受用户质量 + 噪声影响，实验组额外加真实效果
noise_control = np.random.normal(0, 0.04, N)
noise_treat   = np.random.normal(0, 0.04, N)

post_ctr_control = user_quality[:N] * 0.15 + 0.08 + noise_control
post_ctr_treat   = user_quality[N:] * 0.15 + 0.08 + TRUE_EFFECT + noise_treat
post_ctr_control = np.clip(post_ctr_control, 0, 1)
post_ctr_treat   = np.clip(post_ctr_treat, 0, 1)

df_control = pd.DataFrame({
    'user_id': [f'U{i:05d}' for i in range(N)],
    'group': 'control',
    'pre_ctr': pre_ctr_control,
    'post_ctr': post_ctr_control,
})
df_treat = pd.DataFrame({
    'user_id': [f'U{i:05d}' for i in range(N, N * 2)],
    'group': 'treatment',
    'pre_ctr': pre_ctr_treat,
    'post_ctr': post_ctr_treat,
})

df = pd.concat([df_control, df_treat], ignore_index=True)
df.to_csv('data/ab_experiment.csv', index=False, encoding='utf-8-sig')
print(f"数据生成完成：{len(df)} 条用户记录")
print(f"对照组 post_ctr 均值: {post_ctr_control.mean():.4f}")
print(f"实验组 post_ctr 均值: {post_ctr_treat.mean():.4f}")
print(f"真实提升: {TRUE_EFFECT:.4f}")
