# 12.13_Interpretation
"""解释 boosting 树: 特征相对重要性。"""
import warnings
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_iris(return_X_y=True)
    gbm = GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=0).fit(X, y)
    imp = gbm.feature_importances_
    names = ["sepal_l", "sepal_w", "petal_l", "petal_w"]
    for n, v in zip(names, imp):
        print(f"  {n}: {v:.3f}")
    print("相对重要性衡量各特征对提升的贡献。")


if __name__ == "__main__":
    main()