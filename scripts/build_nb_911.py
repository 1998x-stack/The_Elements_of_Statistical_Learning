"""构建 9.11 深讲笔记本(自举). 用法: python scripts/build_nb_911.py"""
import pathlib
import nbformat as nbf

md = [
    "# 9.11 自举 (Bootstrap)（深讲）",
    "用有效回抽样的重采样估计统计量的分布。",
    "## 1. 重采样",
    "从 N 个样本有放回抽 N 个构成一个重样本。",
    "## 2. 估计标准误",
    "对回归系数重复拟合并画分布直方图。",
]
code = [
    "import numpy as np\nfrom sklearn.linear_model import LinearRegression\n"
    "rng=np.random.default_rng(0)\nn=100\nX=rng.normal(size=(n,3))\ny=X@np.array([1.0,-2.0,0.5])+0.4*rng.standard_normal(n)\n"
    "betas=[]\nfor _ in range(500):\n"
    "    idx=rng.integers(0,n,n)\n    betas.append(LinearRegression().fit(X[idx],y[idx]).coef_)\n"
    "betas=np.array(betas)\nprint('自举标准误:', np.round(betas.std(axis=0),3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "plt.hist(betas[:,0],bins=40)\nplt.xlabel('beta_1 自举分布'); plt.title('Bootstrap 分布')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("09_Model_Assessment_and_Selection/9.11_Bootstrap_Methods/9.11_Bootstrap_Methods.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)