"""构建 11.4 深讲笔记本(MARS). 用法: python scripts/build_nb_114.py"""
import pathlib
import nbformat as nbf

md = [
    "# 11.4 MARS（深讲）",
    "用 hinge 基前向逐步拟合一维分段线性函数。",
    "## 1. 候选基",
    "$(x-\\xi)_+ = \\max(0, x-\\xi)$。",
    "## 2. 前向选择",
    "每次加入使残差平方和最小的 hinge 再线性拟合。",
]
code = [
    "import numpy as np\nfrom sklearn.linear_model import LinearRegression\n"
    "rng=np.random.default_rng(0)\n"
    "x=np.sort(rng.uniform(0,1,200))\n"
    "y=2+5*np.maximum(x-0.4,0)-3*np.maximum(x-0.7,0)+0.1*rng.standard_normal(200)\n"
    "def hinge(a,k): return np.maximum(a-k,0)\n"
    "K=[]\n"
    "for _ in range(4):\n"
    "    best=None\n"
    "    for k in x:\n"
    "        B=np.column_stack([np.ones_like(x)]+[hinge(x,kk) for kk in K+[k]])\n"
    "        m=LinearRegression().fit(B,y)\n"
    "        rss=np.sum((y-m.predict(B))**2)\n"
    "        if best is None or rss<best[0]: best=(rss,k)\n"
    "    K.append(best[1])\n"
    "B=np.column_stack([np.ones_like(x)]+[hinge(x,k) for k in K])\n"
    "m=LinearRegression().fit(B,y)\n"
    "print('选中折点', np.round(K,3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['SimHei']\n"
    "xx=np.linspace(0,1,300)\n"
    "Bx=np.column_stack([np.ones_like(xx)]+[hinge(xx,k) for k in K])\n"
    "plt.scatter(x,y,s=6,alpha=0.5)\nplt.plot(xx,m.predict(Bx),lw=2)\nplt.title('MARS 分段线性拟合')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("11_Additive_Models,_Trees,_and_Related_Methods/11.4_MARS:_Multivariate_Adaptive_Regression_Splines/11.4_MARS:_Multivariate_Adaptive_Regression_Splines.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)