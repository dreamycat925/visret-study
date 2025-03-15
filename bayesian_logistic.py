import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# 🔹 NumPy のランダムシードを設定（PyMC のランダム性にも影響）
np.random.seed(42)

# ✅ AP群を除外し、HP vs AD の比較を行う
Z = df_com[df_com['group'].isin(['hs', 'ci'])].copy()

# ✅ 説明変数（カテゴリ変数をダミー化）
X = Z[['検査時の年齢', '性別', '教育歴', '絵の再認課題_点数', '絵の再認課題_虚再認の数']]
X = pd.get_dummies(X, columns=['性別'], drop_first=True, dtype=int)  

# ✅ 目的変数（AD = 1, HP = 0）
y = (Z['group'] == 'ci').astype(int).values  

# 標準化
num_cols = ['検査時の年齢', '教育歴', '絵の再認課題_点数', '絵の再認課題_虚再認の数']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ✅ PyMCによるベイズロジスティック回帰
with pm.Model() as bayesian_logit_model:
    β_age = pm.Normal("β_age", mu=0, sigma=1)
    β_edu = pm.Normal("β_edu", mu=0, sigma=1)
    β_gender = pm.Normal("β_gender", mu=0, sigma=1)
    β_score = pm.Normal("β_score", mu=0, sigma=1)
    β_fal_recog = pm.Normal("β_fal_recog", mu=0, sigma=1)
    intercept = pm.Normal("intercept", mu=0, sigma=2)
    
    # 線形結合
    μ = (
        intercept
        + β_age * X["検査時の年齢"].values
        + β_edu * X["教育歴"].values
        + β_gender * X["性別_男"].values
        + β_score * X['絵の再認課題_点数'].values
        + β_fal_recog * X['絵の再認課題_虚再認の数'].values
    )
    
    # シグモイド関数で確率を得る
    p = pm.math.sigmoid(μ)
    
    # ベルヌーイ分布の尤度
    likelihood = pm.Bernoulli("y", p=p, observed=y)
    
    # ✅ MCMCサンプリング（エラーを防ぐため、サンプル数を調整）
    trace_bayes_logit = pm.sample(2000, tune=1000, target_accept=0.98, return_inferencedata=True)

# ✅ 事後分布の要約
summary = az.summary(trace_bayes_logit, stat_funcs={"median": np.median}, hdi_prob=0.95)
print(summary)

# ✅ オッズ比を計算
odds_ratios = np.exp(summary['median'])
print("\nオッズ比:")
print(odds_ratios)

odds_ratios_lower = np.exp(summary['hdi_2.5%'])
print("\nオッズ比下限:")
print(odds_ratios_lower)

odds_ratios_upper = np.exp(summary['hdi_97.5%'])
print("\nオッズ比上限:")
print(odds_ratios_upper)

# ✅ 事後確率を計算
def compute_posterior_probabilities(trace, param):
    """事後確率を求める"""
    samples = trace.posterior[param].values.flatten()
    prob_positive = (samples > 0).mean()
    prob_negative = (samples < 0).mean()
    return prob_positive, prob_negative

print("\n事後確率:")
for param in ["β_age", "β_edu", "β_gender", "β_score", "β_fal_recog"]:
    p_pos, p_neg = compute_posterior_probabilities(trace_bayes_logit, param)
    print(f"{param}: P(β > 0) = {p_pos:.3f}, P(β < 0) = {p_neg:.3f}")

# ✅ 予測確率を取得（エラーを修正）
with bayesian_logit_model:
    posterior_pred = pm.sample_posterior_predictive(trace_bayes_logit, var_names=["y"])

# ✅ 事後分布の平均を計算（エラーを防ぐため、`posterior_predictive` の構造を確認）
pred_prob = posterior_pred.posterior_predictive["y"].mean(dim=["chain", "draw"]).values  

# ✅ ROC曲線の作成
fpr, tpr, thresholds = roc_curve(y, pred_prob)  
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc}')

# ✅ プロット
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')

# 目盛りのフォントサイズと太さを調整
plt.tick_params(axis='both', which='major', labelsize=22, width=2)

# 目盛りラベルを太字にする
for label in plt.gca().get_xticklabels():
    label.set_fontweight('bold')
for label in plt.gca().get_yticklabels():
    label.set_fontweight('bold')
    
plt.legend()
plt.tight_layout()
plt.show()
