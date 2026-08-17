"""构建 7.4 深讲笔记本(平滑样条). 用法: python scripts/build_nb_74.py"""
import pathlib
import nbformat as nbf

md = [
    "# 7.4 平滑样条（深讲）",
    "展示罚函数 $\\lambda\\int (f'')^2$ 如何控制平滑度。",
    "## 1. 目标",
    "$$\\min_f \\sum (y_i-f(x_i))^2+\\lambda\\int (f'')^2$$",
    "## 2. 光滑度",
    "不同 $\\lambda$（/s）给出不同偏差-方差取舍。",
]
code = [
    "import numpy as np\nfrom scipy.interpolate import UnivariateSpline\n"
    "rng=np.random.default_rng(0)\n"
    "x=np.sort(rng.uniform(0,1,100)); y=np.sin(6*x)+0.2*rng.standard_normal(100)\n"
    "xx=np.linspace(0,1,300)\n"
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "plt.scatter(x,y,s=6,alpha=0.4)\n"
    "for s in (0.1,0.6): plt.plot(xx, UnivariateSpline(x,y,s=s)(xx), label=f's={s}')\n"
    "plt.plot(xx,np.sin(6*xx),'k--',label='真实')\nplt.legend(); plt.title('平滑样条')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("07_Basis_Expansions_and_Regularization/7.4_Smoothing_Splines/7.4_Smoothing_Splines.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)