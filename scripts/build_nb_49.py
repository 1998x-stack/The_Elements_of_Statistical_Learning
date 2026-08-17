"""构建 4.9 深讲笔记本(偏差-方差). 用法: python scripts/build_nb_49.py"""
import pathlib
import nbformat as nbf

md = [
    "# 4.9 模型选择与偏差-方差权衡（深讲）",
    "用最近邻回归模拟偏差-方差分解：不同 $k$ 下偏差²与方差的变化。",
    "## 1. 设定与实验",
    "对正弦回归目标 $f(x)=\\sin(1.5x)$ 加噪声，固定测试网格，多组重复拟合 kNN 得到偏差²与方差。",
]
code = [
    "import numpy as np\nfrom sklearn.neighbors import KNeighborsRegressor\n"
    "rng=np.random.default_rng(0)\n"
    "def f(x): return np.sin(1.5*x)\n"
    "n,rep=120,150\nks=[1,3,9,30,80]\nxs=np.linspace(0,4,150)\ntrue=f(xs)\nout=[]\n"
    "for k in ks:\n"
    "    P=np.zeros((rep,len(xs)))\n"
    "    for r in range(rep):\n"
    "        x=rng.uniform(0,4,n); y=f(x)+0.2*rng.standard_normal(n)\n"
    "        P[r]=KNeighborsRegressor(n_neighbors=k).fit(x[:,None],y).predict(xs[:,None])\n"
    "    out.append((k, np.mean((P.mean(0)-true)**2), np.mean(np.var(P,axis=0))))\n"
    "print('k  bias2  var')\nfor k,b,v in out: print(k, round(b,3), round(v,3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "ks=[o[0] for o in out]; b2=[o[1] for o in out]; v=[o[2] for o in out]\n"
    "plt.plot(ks,b2,'o-',label='bias^2'); plt.plot(ks,v,'s-',label='var')\n"
    "plt.xlabel('k'); plt.legend(); plt.title('偏差-方差权衡 \\u0028kNNN\\u0029')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("04_Overview_of_Supervised_Learning/4.9_Model_Selection_and_the_Bias–Variance_Tradeoff/4.9_Model_Selection_and_the_Bias–Variance_Tradeoff.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)