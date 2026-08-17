# 8.11_Exercises
"""习题: 用交叉验证为高斯核平滑选择带宽。"""
import warnings
import numpy as np
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def gauss_smooth(x, y, xx, h):
    out = np.empty_like(xx)
    for j, t in enumerate(xx):
        w = np.exp(-((x - t) / h) ** 2)
        out[j] = np.sum(w * y) / np.sum(w)
    return out


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 200))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(200)
    hs = np.linspace(0.03, 0.5, 40)
    kf = KFold(5, shuffle=True, random_state=0)
    cv = []
    for h in hs:
        e = 0.0
        for tr, va in kf.split(x):
            e += np.mean((y[va] - gauss_smooth(x[tr], y[tr], x[va], h)) ** 2)
        cv.append(e / 5)
    h_cv = hs[int(np.argmin(cv))]
    # 用真实函数评估各带宽的泛化误差作对照
    truth = []
    xx = np.linspace(0, 1, 200)
    for h in hs:
        truth.append(np.mean((np.sin(6 * xx) - gauss_smooth(x, y, xx, h)) ** 2))
    h_true = hs[int(np.argmin(truth))]
    print("CV 最优带宽 h_cv =", round(h_cv, 3), "| 真实最优 h =", round(h_true, 3))
    print("说明: CV 通常能选到接近真实最优的带宽。")


if __name__ == "__main__":
    main()