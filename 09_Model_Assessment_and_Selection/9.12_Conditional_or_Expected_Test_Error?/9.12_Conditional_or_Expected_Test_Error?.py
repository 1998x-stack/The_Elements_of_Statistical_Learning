# 9.12_Conditional_or_Expected_Test_Error?
"""有条件的 vs 期望测试误差: 固定训练集与跨训练集期望的对比。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    beta = np.array([1.0, -1.0])
    n = 80
    Xtr = rng.normal(size=(n, 2))
    ytr = Xtr @ beta + 0.5 * rng.standard_normal(n)
    lr = LinearRegression().fit(Xtr, ytr)
    Xte = rng.normal(size=(2000, 2))
    yte = Xte @ beta + 0.5 * rng.standard_normal(2000)
    cond = np.mean((yte - lr.predict(Xte)) ** 2)  # 条件(固定训练集)
    errs = []
    for _ in range(100):
        Xr = rng.normal(size=(n, 2))
        yr = Xr @ beta + 0.5 * rng.standard_normal(n)
        errs.append(np.mean((yte - LinearRegression().fit(Xr, yr).predict(Xte)) ** 2))
    print(f"条件测试误差: {cond:.3f}   期望测试误差: {np.mean(errs):.3f}")


if __name__ == "__main__":
    main()