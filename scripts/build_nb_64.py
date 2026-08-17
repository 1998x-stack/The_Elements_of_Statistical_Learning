"""构建 6.4 深讲笔记本(逻辑回归). 用法: python scripts/build_nb_64.py"""
import pathlib
import nbformat as nbf

md = [
    "# 6.4 逻辑回归（深讲）",
    "对后验概率建模并绘制决策面。",
    "## 1. 模型",
    "$$\\log\\frac{p}{1-p}=\\beta_0+x^T\\beta$$",
    "## 2. 决策边界",
    "在两类合成数据上可视化决策区域；注意边界仍是线性。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import make_blobs\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "X,y=make_blobs(n_samples=300,centers=2,cluster_std=2.0,random_state=0)\n"
    "clf=LogisticRegression().fit(X,y)\nprint('准确率', round(clf.score(X,y),3), '系数', np.round(clf.coef_[0],2))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "plt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "gx,gy=np.meshgrid(np.linspace(X[:,0].min()-2,X[:,0].max()+2,300),np.linspace(X[:,1].min()-2,X[:,1].max()+2,300))\n"
    "pp=clf.predict_proba(np.c_[gx.ravel(),gy.ravel()])[:,1].reshape(gx.shape)\n"
    "plt.contourf(gx,gy,pp,alpha=0.5,cmap='RdBu')\nplt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',s=12)\nplt.colorbar()\nplt.title('逻辑回归概率')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("06_Linear_Methods_for_Classification/6.4_Logistic_Regression/6.4_Logistic_Regression.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)