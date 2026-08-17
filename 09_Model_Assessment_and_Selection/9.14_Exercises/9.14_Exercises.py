# 9.14_Exercises
"""习题: 对比 LOOCV 与 K 折 CV 对线性回归误差的估计。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 60
    x = np.sort(rng.uniform(-3, 3, n))
    y = np.sin(x) + 0.3 * rng.standard_normal(n)
    X = np.c_[x, x ** 2, x ** 3]
    # LOOCV
    loo = LeaveOneOut()
    e = 0.0
    for tr, va in loo.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        e += (y[va][0] - m.predict(X[va])[0]) ** 2
    loo_err = e / n
    # K 折
    kf = KFold(10, shuffle=True, random_state=0)
    kf_err = 0.0
    for tr, va in kf.split(X):
        kf_err += mean_squared_error(y[va], LinearRegression().fit(X[tr], y[tr]).predict(X[va]))
    kf_err /= 10
    print(f"LOOCV 误差 ≈ {loo_err:.4f}    K=10 折误差 ≈ {kf_err:.4f}")
    print("两者通常接近; LOOCV 几乎无偏但方差较大。")


if __name__ == "__main__":
    main()