import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 説明変数
X = df_com[df_com['group'] == 'hs'][['検査時の年齢', '性別', '教育歴', '絵の再認課題_点数', '絵の再認課題_虚再認の数']].copy()

# カテゴリ変数のエンコーディング
X = pd.get_dummies(X, columns=['性別'], drop_first=True, dtype=int) 

# 目的変数
y = (df_com['group'] == 'ci').astype(int).values  

# 標準化
scaler = StandardScaler()
num_cols = ['検査時の年齢', '教育歴', '絵の再認課題_点数', '絵の再認課題_虚再認の数']  # 標準化対象の数値カラム
X[num_cols] = scaler.fit_transform(X[num_cols])

# ✅ ベイズ線形回帰モデルの構築（標準化後のデータを使用）
with pm.Model() as model:
    beta_0 = pm.Normal("beta_0", mu=0, sigma=1)
    beta_edu = pm.Normal("beta_edu", mu=0, sigma=1)
    beta_gender = pm.Normal("beta_gender", mu=0, sigma=1)
    beta_false_recog = pm.Normal("beta_false_recog", mu=0, sigma=1)
    beta_age = pm.Normal("beta_age", mu=0, sigma=1)

    # ✅ 回帰式（すべてのデータに同じ係数を適用）
    mu = (
        beta_0
        + beta_age * X["検査時の年齢"]
        + beta_edu * X["教育歴"]
        + beta_gender * X["性別_男"]  # 性別はスケーリングしていない
        + beta_false_recog * X["絵の再認課題_虚再認の数"]
    )

    sigma = pm.HalfNormal("sigma", sigma=1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=X["絵の再認課題_点数"])

    # ✅ MCMCサンプリング
    trace = pm.sample(2000, tune=1000, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

# ✅ 事後分布の可視化
az.plot_posterior(trace, var_names=["beta_false_recog", "beta_age", "beta_edu", "beta_gender"])
plt.show()

# 事後分布の要約
summary = az.summary(trace, stat_funcs={"median": np.median}, hdi_prob=0.95)
print(summary)

print('')

# 事後確率を計算
def compute_posterior_probabilities(trace, param):
    """事後確率を求める"""
    samples = trace.posterior[param].values.flatten()
    prob_positive = (samples > 0).mean()
    prob_negative = (samples < 0).mean()
    return prob_positive, prob_negative

print("\n事後確率:")
for param in ["beta_age", "beta_edu", "beta_gender", "beta_false_recog"]:
    p_pos, p_neg = compute_posterior_probabilities(trace, param)
    print(f"{param}: P(β > 0) = {p_pos:.3f}, P(β < 0) = {p_neg:.3f}")

def memory_score(score, edu, false_recog, trace):
    return score + (az.summary(trace, stat_funcs={"median": np.median}).loc['beta_edu', 'median'] * edu) + (az.summary(trace, stat_funcs={"median": np.median}).loc['beta_false_recog', 'median'] * false_recog)
