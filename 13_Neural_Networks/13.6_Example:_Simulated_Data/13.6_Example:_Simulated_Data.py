# 13.6_Example:_Simulated_Data
"""合成数据示例: MLP 在两类交错的合成数据上的拟合。"""
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    print("隐单元数  测试准确率")
    for h in (3, 10, 30):
        m = MLPClassifier(hidden_layer_sizes=(h,), activation="tanh", max_iter=800,
                          random_state=0).fit(Xtr, ytr)
        print(f"  {h:4d}     {m.score(Xte, yte):.3f}")
    print("更宽隐层提升表达但更易过拟合。")


if __name__ == "__main__":
    main()