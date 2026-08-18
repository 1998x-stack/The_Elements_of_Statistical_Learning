# 13.12_Exercises
"""习题: 隐单元数与最大迭代对 MLP 在合成数据测试误差的影响。"""
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    print("隐单元  max_iter  测试准确率")
    for h in (4, 16):
        for mi in (100, 400):
            m = MLPClassifier(hidden_layer_sizes=(h,), max_iter=mi, random_state=0).fit(Xtr, ytr)
            print(f"  {h:3d}    {mi:6d}     {accuracy_score(yte, m.predict(Xte)):.3f}")


if __name__ == "__main__":
    main()