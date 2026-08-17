"""构建 10.5 深讲笔记本(EM). 用法: python scripts/build_nb_105.py"""
import pathlib
import nbformat as nbf

md = [
    "# 10.5 EM 算法（深讲）",
    "手写高斯混合的 EM，观察对数似然单调递增。",
    "## 1. E 步与 M 步",
    "E: 计算责任; M: 用责任加权更新均值/方差/混合比例。",
]
code = [
    "import numpy as np\nfrom scipy.stats import norm\nrng=np.random.default_rng(0)\n"
    "x=np.concatenate([rng.normal(0,1,200),rng.normal(4,1.5,200)])\n"
    "m1,m2,s1,s2,pi=1.0,3.0,1.0,1.0,0.5\n"
    "def loglik(x,m1,m2,s1,s2,pi):\n"
    "    return np.sum(np.log(pi*norm.pdf(x,m1,s1)+(1-pi)*norm.pdf(x,m2,s2)+1e-12))\n"
    "print('iter  loglik')\n"
    "for it in range(30):\n"
    "    p1=pi*norm.pdf(x,m1,s1); p2=(1-pi)*norm.pdf(x,m2,s2); g=p1/(p1+p2+1e-12)\n"
    "    m1=(g*x).sum()/g.sum(); m2=((1-g)*x).sum()/(1-g).sum()\n"
    "    s1=np.sqrt((g*(x-m1)**2).sum()/g.sum()); s2=np.sqrt(((1-g)*(x-m2)**2).sum()/(1-g).sum())\n"
    "    pi=g.mean()\n"
    "    if it%5==0: print(it, round(loglik(x,m1,m2,s1,s2,pi),3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "gx=np.linspace(-4,9,300)\n"
    "fit=pi*norm.pdf(gx,m1,s1)+(1-pi)*norm.pdf(gx,m2,s2)\n"
    "plt.hist(x,bins=40,density=True,alpha=0.4)\nplt.plot(gx,fit,'r')\nplt.title('EM 拟合的高斯混合密度')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("10_Model_Inference_and_Averaging/10.5_The_EM_Algorithm/10.5_The_EM_Algorithm.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)