# 13.8_Discussion
"""讨论: MLP vs 线性/核在 digits 上的简要对比。"""
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    models = {
        "logistic": LogisticRegression(max_iter=500).fit(Xtr, ytr),
        "MLP": MLPClassifier(hidden_layer_sizes=(30,), max_iter=300, random_state=0).fit(Xtr, ytr),
        "RBF-SVM": SVC(kernel="rbf").fit(Xtr, ytr),
    }
    for n, m in models.items():
        print(f"{n}: 测试准确率={accuracy_score(yte, m.predict(Xte)):.3f}")


if __name__ == "__main__":
    main()