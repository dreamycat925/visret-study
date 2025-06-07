# 　虚再認の数とgroup　有意のみ

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# 説明変数
X = df_com.copy()
X = X[X['group'] != 'ci']
X = X[['検査時の年齢', '教育歴', '絵の再認課題_点数', '絵の再認課題_虚再認の数']]

# ✅ グループを数値化（0=健常, 1=患者）
X["group"] = (df_com["group"] == "ap").astype(int)

# ✅ ベイズ線形回帰（点数の群間差を評価）
with pm.Model() as model_score_adj:
    beta_0 = pm.Normal("beta_0", mu=0, sigma=1)  # 切片
    beta_group = pm.Normal("beta_group", mu=0, sigma=1)  # 群の影響
    beta_age = pm.Normal("beta_age", mu=0, sigma=1)  # 年齢の影響
    beta_score = pm.Normal("beta_score", mu=0, sigma=1)  # スコアの影響

    mu = (
        beta_0
        + beta_group * X["group"]
        + beta_age * X["検査時の年齢"]
        + beta_score * X["絵の再認課題_点数"]
    )

    sigma = pm.HalfNormal("sigma", sigma=1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=X["絵の再認課題_虚再認の数"])

    # ✅ MCMC サンプリング
    trace_score_adj = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

# 事後分布の要約
summary = az.summary(trace_score_adj, stat_funcs={"median": np.median}, hdi_prob=0.95, round_to=4)
print(summary)

# 事後確率を計算
def compute_posterior_probabilities(trace, param):
    """事後確率を求める"""
    samples = trace.posterior[param].values.flatten()
    prob_positive = (samples > 0).mean()
    prob_negative = (samples < 0).mean()
    return prob_positive, prob_negative

print("\n事後確率:")
for param in ["beta_group", "beta_age", "beta_score"]:
    p_pos, p_neg = compute_posterior_probabilities(trace_score_adj, param)
    print(f"{param}: P(β > 0) = {p_pos:.3f}, P(β < 0) = {p_neg:.3f}")

az.plot_trace(trace_score_adj, figsize=(20, 12), combined=False, compact=False)
plt.tight_layout()
az.plot_posterior(trace_score_adj, hdi_prob=0.95)
plt.show()