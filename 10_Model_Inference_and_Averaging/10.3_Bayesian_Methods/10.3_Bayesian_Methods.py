# 10.3_Bayesian_Methods
"""贝叶斯方法: 高斯均值/正态似然的共轭后验。"""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 50
    mu, sigma = 1.0, 1.0
    x = rng.normal(mu, sigma, n)
    # 先验 N(m0, tau0^2); 似然方差已知
    m0, tau0 = 0.0, 10.0
    prior_prec = 1.0 / tau0 ** 2
    data_prec = n / sigma ** 2
    post_var = 1.0 / (prior_prec + data_prec)
    post_mean = post_var * (prior_prec * m0 + data_prec * x.mean())
    mle = x.mean()
    print(f"MLE 均值 = {mle:.3f}    贝叶斯后验均值 = {post_mean:.3f}")
    print(f"后验方差 = {post_var:.4f}  (先验把估计拉向 m0)")


if __name__ == "__main__":
    main()