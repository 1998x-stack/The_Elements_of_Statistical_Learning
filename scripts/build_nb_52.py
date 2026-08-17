"""构建 5.2 深讲笔记本(最小二乘). 用法: python scripts/build_nb_52.py"""
import pathlib
import nbformat as nbf

md = [
    "# 5.2 线性回归与最小二乘（深讲）",
    "本笔记本用糖尿病数据逐步实现正规方程解 $\\hat\\beta=(X^TX)^{-1}X^Ty$。",
    "## 1. 数据与设计矩阵",
    "加载数据，加一列截距。",
    "## 2. 闭式求解",
    "用 `np.linalg.solve` 求解线性方程组，避免直接求逆，数值更稳。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import load_diabetes\n"
    "X, y = load_diabetes(return_X_y=True)\nXb = np.c_[np.ones(len(X)), X]\nprint('设计矩阵形状:', Xb.shape)",
    "beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)\n"
    "rss = np.sum((y - Xb@beta)**2)\nprint('beta0 =', round(beta[0],3), ' RSS =', round(rss,2))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "plt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "yhat = Xb@beta\nplt.scatter(yhat, y-yhat, s=10)\nplt.axhline(0,c='k',lw=0.8)\n"
    "plt.xlabel('拟合值'); plt.ylabel('残差'); plt.title('残差-拟合图')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("05_Linear_Methods_for_Regression/5.2_Linear_Regression_Models_and_Least_Squares/5.2_Linear_Regression_Models_and_Least_Squares.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)