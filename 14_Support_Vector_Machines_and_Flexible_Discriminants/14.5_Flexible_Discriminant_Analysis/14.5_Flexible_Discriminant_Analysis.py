# 14.5_Flexible_Discriminant_Analysis
"""灵活判别分析(FDA): 特征映射(样条/多项式)后做 LDA 得非线性边界。"""
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_moons(n_samples=250, noise=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    # FDA 思想: 多项式特征 + LDA
    PF = PolynomialFeatures(3, include_bias=False)
    Ztr, Zte = PF.fit_transform(Xtr), PF.transform(Xte)
    lda = LinearDiscriminantAnalysis().fit(Ztr, ytr)
    print("FDA(多项式映射+LDA) 测试准确率 =", round(accuracy_score(yte, lda.predict(Zte)), 3))
    print("FDA: 先在扩展特征空间做 LDA, 得到灵活非线性边界。")


if __name__ == "__main__":
    main()