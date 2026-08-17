"""构建 7.2 深讲笔记本(样条). 用法: python scripts/build_nb_72.py"""
import pathlib
import nbformat as nbf

md = [
    "# 7.2 分段多项式与样条（深讲）",
    "用截断幂基构造三次样条，并与全局高次多项式比较。",
    "## 1. 截断幂基",
    "$$(x-\\xi)_+^3=\\max(0,(x-\\xi))^3$$ 在节点处引入二阶光滑的连接。",
    "## 2. 拟合曲线",
    "比较样条与 30 次多项式的行为。",
]
code = [
    "import numpy as np\nrng=np.random.default_rng(0)\n"
    "x=np.sort(rng.uniform(0,1,120))\ny=np.sin(4*x)+0.15*rng.standard_normal(120)\n"
    "knots=[0.4,0.7]\n"
    "def trunc(X,k): return np.maximum(X-k,0)**3\n"
    "def basis(X):\n"
    "    X=np.asarray(X)[:,None]\n"
    "    B=[np.ones_like(X[:,0]),X[:,0],X[:,0]**2,X[:,0]**3]\n"
    "    for k in knots: B.append(trunc(X[:,0],k))\n"
    "    return np.column_stack(B)\nprint('基矩阵形状', basis(x).shape)",
    "from sklearn.linear_model import LinearRegression\nm=LinearRegression().fit(basis(x),y)\nxx=np.linspace(0,1,200)\n"
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "plt.scatter(x,y,s=8,alpha=0.4)\nplt.plot(xx,m.predict(basis(xx)),label='三次样条')\nplt.legend(); plt.title('样条基拟合')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("07_Basis_Expansions_and_Regularization/7.2_Piecewise_Polynomials_and_Splines/7.2_Piecewise_Polynomials_and_Splines.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)