# 14.4_Generalizing_Linear_Discriminant_Analysis
"""推广 LDA: 用降秩判别空间(LDA eigen solver)。"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_iris(return_X_y=True)
    lda = LinearDiscriminantAnalysis(solver="eigen").fit(X, y)
    print("判别方向数(降秩) =", len(lda.coef_))
    print("训练准确率 =", round(lda.score(X, y), 3))
    print("LDA 通过判别空间(low-rank)把数据投影分层。")


if __name__ == "__main__":
    main()