"""构建 5.4 深讲笔记本(收缩方法). 用法: python scripts/build_nb_54.py"""
import pathlib
import nbformat as nbf

md = [
    "# 5.4 收缩方法: 岭与 Lasso（深讲）",
    "比较 $\\ell_2$（岭）与 $\\ell_1$（Lasso）罚项对系数的影响。",
    "## 1. 岭回归存在的闭式解",
    "$$\\hat\\beta^{ridge}=(X^TX+\\lambda I)^{-1}X^Ty$$",
    "## 2. 系数路径与稀疏性",
    "观察随 $\\lambda$ 增大岭系数连续缩小；Lasso 使部分系数精确为零。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import make_regression\nfrom sklearn.linear_model import Ridge\n"
    "X,y=make_regression(n_samples=80,n_features=8,n_informative=5,noise=0.5,random_state=0)\n"
    "al=np.logspace(-3,3,30)\nC=np.array([Ridge(alpha=a).fit(X,y).coef_ for a in al])\nprint('岭系数矩阵形状', C.shape)",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "plt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "plt.plot(np.log10(al), C)\nplt.xlabel('log10(lambda)'); plt.ylabel('系数')\n"
    "plt.title('岭回归系数路径'); plt.show()",
    "from sklearn.linear_model import Lasso\n"
    "lasso=Lasso(alpha=0.5,max_iter=5000).fit(X,y)\n"
    "print('Lasso 非零系数个数', int(np.sum(lasso.coef_!=0)))",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("05_Linear_Methods_for_Regression/5.4_Shrinkage_Methods/5.4_Shrinkage_Methods.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)