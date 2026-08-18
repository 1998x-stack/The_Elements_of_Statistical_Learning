# 14.7_Mixture_Discriminant_Analysis
"""混合判别分析(MDA): 每类用个高斯混合(子类) + 贝叶斯。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    # 三类, 每类两团(子类) -> 用每类的 GMM 拟合
    X, y = make_blobs(n_samples=300, centers=6, cluster_std=0.8, random_state=0)
    y = (y // 2).astype(int)   # 6 团合并为 3 类
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    gms = {k: GaussianMixture(n_components=2, random_state=0).fit(Xtr[ytr == k])
           for k in np.unique(ytr)}
    prob = np.column_stack([g.score_samples(Xte) for g in [gms[k] for k in np.unique(ytr)]])
    pred = np.unique(ytr)[prob.argmax(axis=1)]
    print("MDA(每类GMM) 测试准确率 =", round(accuracy_score(yte, pred), 3))
    print("MDA: 每类允许多个子类中心, 更灵活的类形状。")


if __name__ == "__main__":
    main()