# 6.2_Linear_Regression_of_an_Indicator_Matrix
"""用指示矩阵做线性回归分类: 对每列回归再取 argmax。"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_iris(return_X_y=True)
    Y0 = (y[:, None] == np.arange(3)).astype(float)  # K 个 0-1 指示列
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y0, stratify=y, test_size=0.3, random_state=0)
    model = LinearRegression().fit(Xtr, Ytr)
    pred = np.argmax(model.predict(Xte), axis=1)
    acc = np.mean(pred == np.argmax(Yte, axis=1))
    print(f"指示矩阵线性回归 分类准确率 = {acc:.3f}")


if __name__ == "__main__":
    main()