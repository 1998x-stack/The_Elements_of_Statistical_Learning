# 16.6_Non_negative_Matrix_Factorization
"""非负矩阵分解(NMF): digits 数据的部分表示。"""
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF
from sklearn.preprocessing import MaxAbsScaler
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, _ = load_digits(return_X_y=True)
    X = MaxAbsScaler().fit_transform(X)
    nmf = NMF(n_components=8, init="nndsvda", random_state=0).fit(X)
    print("NMF 负载(basis)形状:", nmf.components_.shape)
    print("重建误差(弗罗比尼乌斯):", round(nmf.reconstruction_err_, 2))
    print("NMF 得到非负因子, 可解释为部分/零件表示。")


if __name__ == "__main__":
    main()