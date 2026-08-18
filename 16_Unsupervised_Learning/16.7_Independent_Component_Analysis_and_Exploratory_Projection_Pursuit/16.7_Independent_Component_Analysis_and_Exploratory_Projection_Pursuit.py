# 16.7_Independent_Component_Analysis_and_Exploratory_Projection_Pursuit
"""独立成分分析(ICA): 分离混合的独立信号。"""
import warnings
import numpy as np
from sklearn.decomposition import FastICA
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    t = np.linspace(0, 8, 2000)
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t + 1))
    S = np.c_[s1, s2].T
    A = np.array([[1.0, 0.7], [0.5, 1.0]])
    X = A @ S            # 混合
    ica = FastICA(n_components=2, random_state=0).fit(X.T)
    S_hat = ica.fit_transform(X.T)
    print("ICA 分离后与原件相关性:", np.round(np.corrcoef(S.T, S_hat)[:2, 2:], 2))
    print("ICA 假设非高斯独立源, 用于盲源分离。")


if __name__ == "__main__":
    main()