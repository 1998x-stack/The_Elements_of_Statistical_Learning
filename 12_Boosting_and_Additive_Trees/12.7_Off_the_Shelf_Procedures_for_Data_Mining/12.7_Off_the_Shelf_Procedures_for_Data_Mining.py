# 12.7_Off_the_Shelf_Procedures_for_Data_Mining
"""现成(off-the-shelf)增强: sklearn GradientBoostingClassifier 在 iris 上。"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                                     random_state=0).fit(Xtr, ytr)
    print("GradientBoosting 测试准确率 =", round(gbm.score(Xte, yte), 3))
    print("现成 GBM 默认常用: 树深2-3、η≈0.1、数百棵树。")


if __name__ == "__main__":
    main()