# 13.7_Example:_ZIP_Code_Data
"""手写数字示例: 用 sklearn digits(代 ZIP) 的 MLP 分类。"""
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_digits(return_X_y=True)   # 8x8 手写数字
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    mlp = MLPClassifier(hidden_layer_sizes=(30,), activation="tanh", max_iter=300,
                        random_state=0).fit(Xtr, ytr)
    print("digits MLP 测试准确率 =", round(accuracy_score(yte, mlp.predict(Xte)), 3))
    print("书中用 ZIP 手写数字; 这里用 8x8 digits 作代理。")


if __name__ == "__main__":
    main()