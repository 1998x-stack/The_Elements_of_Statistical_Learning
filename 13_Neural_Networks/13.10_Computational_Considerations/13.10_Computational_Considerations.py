# 13.10_Computational_Considerations
"""神经网络计算: 不同样本量下 MLP 拟合耗时。"""
import time
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    for n in (1000, 5000):
        X, y = make_classification(n_samples=n, n_features=20, random_state=0)
        t0 = time.perf_counter()
        MLPClassifier(hidden_layer_sizes=(30,), max_iter=100, random_state=0).fit(X, y)
        print(f"n={n}: 拟合耗时 {time.perf_counter()-t0:.2f}s")
    print("训练随数据与层数增长; 用批处理/GPU 加速。")


if __name__ == "__main__":
    main()