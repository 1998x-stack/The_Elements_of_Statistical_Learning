# 9.2_Bias,_Variance_and_Model_Complexity
"""偏差、方差与模型复杂度: 随度数变化的训练/测试误差(U型)。"""
import warnings
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(-3, 3, 200))
    y = np.sin(x) + 0.2 * rng.standard_normal(200)
    Xtr, Xte, ytr, yte = train_test_split(x, y, test_size=0.4, random_state=0)
    degs = range(1, 16)
    print("degree  in-sample  test")
    best = (None, np.inf)
    for d in degs:
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(Xtr[:, None], ytr)
        ins = mean_squared_error(ytr, m.predict(Xtr[:, None]))
        tes = mean_squared_error(yte, m.predict(Xte[:, None]))
        if tes < best[1]:
            best = (d, tes)
        if d % 3 == 0 or d in (1, 2):
            print(f"{d:3d}   {ins:.3f}   {tes:.3f}")
    print("测试误差最低度数:", best[0])


if __name__ == "__main__":
    main()