# 5.6_Discussion:_A_Comparison_of_the_Selection_and_Shrinkage_Methods
"""比较子集/岭/Lasso/PCR: 系数数与测试MSE。"""
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    ols = LinearRegression().fit(Xtr, ytr)
    res = [("OLS", np.sum(ols.coef_ != 0), mean_squared_error(yte, ols.predict(Xte)))]
    ridge = Ridge(alpha=8).fit(Xtr, ytr)
    res.append(("RIDGE", np.sum(np.abs(ridge.coef_) > 1e-8), mean_squared_error(yte, ridge.predict(Xte))))
    lasso = Lasso(alpha=1.0, max_iter=5000).fit(Xtr, ytr)
    res.append(("LASSO", np.sum(lasso.coef_ != 0), mean_squared_error(yte, lasso.predict(Xte))))
    pcr = PCA(n_components=6).fit(Xtr)
    m = LinearRegression().fit(pcr.transform(Xtr), ytr)
    res.append(("PCR (6)", 6, mean_squared_error(yte, m.predict(pcr.transform(Xte)))))
    for name, nonz, mse in res:
        print(f"{name:9s} 非零系数={nonz:3d} 测试MSE={mse:.2f}")


if __name__ == "__main__":
    main()