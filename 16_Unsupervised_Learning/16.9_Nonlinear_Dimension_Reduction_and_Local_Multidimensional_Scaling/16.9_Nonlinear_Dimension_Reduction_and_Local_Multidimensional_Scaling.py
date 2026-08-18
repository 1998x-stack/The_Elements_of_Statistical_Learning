# 16.9_Nonlinear_Dimension_Reduction_and_Local_Multidimensional_Scaling
"""非线性降维: Isomap 与局部线性嵌入(LLE)比较。"""
import warnings
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, LocallyLinearEmbedding
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, _ = make_swiss_roll(n_samples=400, noise=0.2, random_state=0)
    iso = Isomap(n_neighbors=12, n_components=2).fit_transform(X)
    lle = LocallyLinearEmbedding(n_neighbors=12, n_components=2).fit_transform(X)
    print("Isomap 嵌入形状:", iso.shape, " LLE 嵌入形状:", lle.shape)
    print("非线性方法(Isomap/LLE/核PCA/局部MDS)处理弯曲低维流形。")


if __name__ == "__main__":
    main()