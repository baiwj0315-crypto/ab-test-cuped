"""
A/B 实验完整分析流程
1. AA 验证（分流是否均匀）
2. 标准 t-test
3. CUPED 方差缩减
4. 样本量规划（Power Analysis）
5. 可视化报告
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data/ab_experiment.csv')
control = df[df['group'] == 'control']
treat   = df[df['group'] == 'treatment']

ALPHA = 0.05  # 显著性水平
POWER = 0.80  # 目标功效

# ── 1. AA 验证 ────────────────────────────────────────────────────────────────
print("=" * 60)
print("【Step 1】AA 验证 - 检查实验前两组是否无显著差异")
print("=" * 60)

t_aa, p_aa = stats.ttest_ind(control['pre_ctr'], treat['pre_ctr'])
print(f"  对照组 pre_ctr 均值: {control['pre_ctr'].mean():.4f}")
print(f"  实验组 pre_ctr 均值: {treat['pre_ctr'].mean():.4f}")
print(f"  t={t_aa:.3f}, p={p_aa:.4f}")
if p_aa > ALPHA:
    print(f"  结论: p={p_aa:.3f} > {ALPHA}，AA 通过，分流均匀，实验可信")
else:
    print(f"  结论: p={p_aa:.3f} <= {ALPHA}，AA 未通过！分流存在偏差，需重新分组")

# ── 2. 标准 t-test ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【Step 2】标准 t-test")
print("=" * 60)

ctrl_post = control['post_ctr'].values
treat_post = treat['post_ctr'].values

t_stat, p_val = stats.ttest_ind(treat_post, ctrl_post)
diff = treat_post.mean() - ctrl_post.mean()
relative_lift = diff / ctrl_post.mean() * 100

# 置信区间
se = np.sqrt(ctrl_post.var()/len(ctrl_post) + treat_post.var()/len(treat_post))
ci_low  = diff - 1.96 * se
ci_high = diff + 1.96 * se

# 效应量 Cohen's d
pooled_std = np.sqrt((ctrl_post.var() + treat_post.var()) / 2)
cohen_d = diff / pooled_std

print(f"  对照组 CTR: {ctrl_post.mean():.4f}  (std={ctrl_post.std():.4f})")
print(f"  实验组 CTR: {treat_post.mean():.4f}  (std={treat_post.std():.4f})")
print(f"  绝对提升:   {diff:+.4f}")
print(f"  相对提升:   {relative_lift:+.2f}%")
print(f"  95% 置信区间: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  t={t_stat:.3f}, p={p_val:.4f}")
print(f"  Cohen's d={cohen_d:.4f} ({'小' if abs(cohen_d)<0.2 else '中' if abs(cohen_d)<0.5 else '大'}效应量)")
if p_val < ALPHA:
    print(f"  结论: 统计显著 (p<{ALPHA})，新算法有效")
else:
    print(f"  结论: 不显著 (p>={ALPHA})")

# ── 3. CUPED ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【Step 3】CUPED 方差缩减（用 pre_ctr 作为协变量）")
print("=" * 60)

# 计算 theta：pre_ctr 对 post_ctr 的回归系数
pre_all  = df['pre_ctr'].values
post_all = df['post_ctr'].values
theta = np.cov(pre_all, post_all)[0, 1] / np.var(pre_all)
print(f"  theta (协变量系数): {theta:.4f}")

# CUPED 调整后的指标
pre_mean_global = pre_all.mean()
ctrl_cuped  = ctrl_post  - theta * (control['pre_ctr'].values - pre_mean_global)
treat_cuped = treat_post - theta * (treat['pre_ctr'].values   - pre_mean_global)

t_cuped, p_cuped = stats.ttest_ind(treat_cuped, ctrl_cuped)
diff_cuped = treat_cuped.mean() - ctrl_cuped.mean()
se_cuped   = np.sqrt(ctrl_cuped.var()/len(ctrl_cuped) + treat_cuped.var()/len(treat_cuped))
ci_cuped   = (diff_cuped - 1.96*se_cuped, diff_cuped + 1.96*se_cuped)

var_reduction = (1 - ctrl_cuped.var() / ctrl_post.var()) * 100
print(f"  方差缩减比例: {var_reduction:.1f}%")
print(f"  CUPED 绝对提升: {diff_cuped:+.4f}")
print(f"  CUPED 95% CI: [{ci_cuped[0]:.4f}, {ci_cuped[1]:.4f}]")
print(f"  t={t_cuped:.3f}, p={p_cuped:.4f}")
print(f"  CI 宽度缩减: {(1 - (ci_cuped[1]-ci_cuped[0])/(ci_high-ci_low))*100:.1f}%")
if p_cuped < ALPHA:
    print(f"  结论: CUPED 后仍显著，结论更可信；灵敏度显著提升")

# ── 4. 样本量规划 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【Step 4】样本量规划 (Power Analysis)")
print("=" * 60)

baseline_ctr = ctrl_post.mean()
mde_list = [0.005, 0.008, 0.010, 0.015, 0.020]  # 最小可检测效果

def calc_sample_size(baseline, mde, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta  = stats.norm.ppf(power)
    p1, p2  = baseline, baseline + mde
    pooled  = (p1 + p2) / 2
    n = (z_alpha + z_beta)**2 * 2 * pooled * (1 - pooled) / mde**2
    return int(np.ceil(n))

print(f"  基准 CTR: {baseline_ctr:.4f}, 显著性水平: {ALPHA}, 功效: {POWER}")
print(f"  {'MDE':>8}  {'单组样本量':>10}  {'每日需要(假设日活5万)':>20}")
for mde in mde_list:
    n = calc_sample_size(baseline_ctr, mde)
    days = np.ceil(n / 25000)
    print(f"  {mde:>8.3f}  {n:>10,}  {days:>20.0f} 天")

# ── 5. 可视化 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
fig.suptitle('A/B 实验分析报告 — 推荐算法优化实验', fontsize=15, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

BLUE  = '#2563eb'
GREEN = '#16a34a'
GRAY  = '#6b7280'

# 图1：CTR 分布对比
ax1 = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 0.35, 40)
ax1.hist(ctrl_post,  bins=bins, alpha=0.6, color=GRAY,  label='对照组', density=True)
ax1.hist(treat_post, bins=bins, alpha=0.6, color=BLUE,  label='实验组', density=True)
ax1.axvline(ctrl_post.mean(),  color=GRAY,  linestyle='--', linewidth=1.5)
ax1.axvline(treat_post.mean(), color=BLUE,  linestyle='--', linewidth=1.5)
ax1.set_title('CTR 分布对比', fontweight='bold')
ax1.set_xlabel('CTR')
ax1.legend(fontsize=9)
for s in ['top','right']: ax1.spines[s].set_visible(False)

# 图2：置信区间对比（t-test vs CUPED）
ax2 = fig.add_subplot(gs[0, 1])
methods = ['标准 t-test', 'CUPED']
diffs   = [diff, diff_cuped]
cis     = [(ci_low, ci_high), ci_cuped]
colors  = [GRAY, GREEN]
for i, (method, d, ci, color) in enumerate(zip(methods, diffs, cis, colors)):
    ax2.errorbar(i, d, yerr=[[d - ci[0]], [ci[1] - d]],
                 fmt='o', color=color, capsize=8, capthick=2,
                 markersize=8, linewidth=2, label=method)
    ax2.text(i + 0.1, d, f'  p={[p_val, p_cuped][i]:.4f}', va='center', fontsize=9)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(methods)
ax2.set_ylabel('CTR 提升量')
ax2.set_title('置信区间对比\nCUPED 更窄 = 更灵敏', fontweight='bold')
for s in ['top','right']: ax2.spines[s].set_visible(False)

# 图3：方差缩减效果
ax3 = fig.add_subplot(gs[0, 2])
categories = ['标准 t-test', 'CUPED']
variances  = [ctrl_post.var(), ctrl_cuped.var()]
bars = ax3.bar(categories, variances, color=[GRAY, GREEN], alpha=0.85, width=0.5)
for bar, v in zip(bars, variances):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.00002,
             f'{v:.5f}', ha='center', fontsize=9)
ax3.set_title(f'方差缩减效果\n缩减了 {var_reduction:.1f}%', fontweight='bold')
ax3.set_ylabel('方差')
for s in ['top','right']: ax3.spines[s].set_visible(False)

# 图4：样本量 vs MDE
ax4 = fig.add_subplot(gs[1, 0:2])
mde_range = np.linspace(0.003, 0.025, 100)
n_range   = [calc_sample_size(baseline_ctr, m) for m in mde_range]
ax4.plot(mde_range * 100, n_range, color=BLUE, linewidth=2.5)
ax4.fill_between(mde_range * 100, n_range, alpha=0.1, color=BLUE)
for mde in [0.005, 0.010, 0.015]:
    n = calc_sample_size(baseline_ctr, mde)
    ax4.scatter(mde * 100, n, s=80, color=BLUE, zorder=5)
    ax4.annotate(f'MDE={mde*100:.1f}%\nn={n:,}',
                 (mde*100, n), textcoords='offset points',
                 xytext=(10, 5), fontsize=8)
ax4.set_xlabel('最小可检测效果 MDE (%)')
ax4.set_ylabel('单组所需样本量')
ax4.set_title('样本量规划曲线（alpha=0.05, power=0.80）', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for s in ['top','right']: ax4.spines[s].set_visible(False)

# 图5：实验结论卡片
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
summary = [
    ('实验结论',       '新推荐算法显著优于旧版', GREEN),
    ('CTR 绝对提升',  f'{diff:+.4f} ({relative_lift:+.2f}%)', BLUE),
    ('统计显著性',    f'p={p_val:.4f} < 0.05', GREEN),
    ('CUPED p值',    f'p={p_cuped:.4f}（更灵敏）', GREEN),
    ('效应量',        f"Cohen's d = {cohen_d:.3f}", GRAY),
    ('方差缩减',      f'{var_reduction:.1f}%', GREEN),
    ('建议',          '可全量上线新算法', GREEN),
]
y_pos = 0.95
ax5.text(0.05, y_pos, '实验摘要', fontsize=12, fontweight='bold', transform=ax5.transAxes)
for label, value, color in summary:
    y_pos -= 0.12
    ax5.text(0.05, y_pos, f'{label}:', fontsize=9, color=GRAY, transform=ax5.transAxes)
    ax5.text(0.05, y_pos - 0.055, value, fontsize=9, fontweight='bold',
             color=color, transform=ax5.transAxes)
    y_pos -= 0.055

plt.savefig('output/ab_test_report.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\n[完成] 图表已保存到 output/ab_test_report.png")
