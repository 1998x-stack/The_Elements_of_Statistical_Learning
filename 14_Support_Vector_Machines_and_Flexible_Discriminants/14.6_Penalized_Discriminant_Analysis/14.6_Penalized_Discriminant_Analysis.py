# 14.6_Penalized_Discriminant_Analysis
"""罚判别分析(PDA): 对特征做岭正则的 LDA, 适合高维。"""
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_classification(n_samples=200, n_features=40, n_informative=10,
                               n_redundant=5, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    print("PDA(收缩/罚 LDA) 测试准确率 =", round(accuracy_score(yte, lda.predict(Xte)), 3))
    print("惩罚(收缩)保持 LDA 在 p 较大/共线时稳定。")


if __name__ == "__main__":
    main()