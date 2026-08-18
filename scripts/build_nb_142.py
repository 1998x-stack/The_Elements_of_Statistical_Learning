"""构建 14.2 深讲笔记本(支持向量分类器). 用法: python scripts/build_nb_142.py"""
import pathlib
import nbformat as nbf

md = [
    "# 14.2 支持向量分类器（深讲）",
    "最大间隔超平面 + 软间隔。",
    "## 1. 模型",
    "$$\\min\\tfrac12\\|\\beta\\|^2+C\\sum\\xi_i$$",
    "## 2. 边界",
    "绘制线性软间隔决策面。",
]
code = [
    "import numpy as np\nfrom sklearn.svm import LinearSVC\nfrom sklearn.datasets import make_blobs\n"
    "X,y=make_blobs(n_samples=120,centers=2,cluster_std=1.5,random_state=0)\n"
    "clf=LinearSVC(C=1.0).fit(X,y)\nprint('准确率', round(clf.score(X,y),3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['SimHei']\n"
    "gx,gy=np.meshgrid(np.linspace(X[:,0].min()-1,X[:,0].max()+1,250),np.linspace(X[:,1].min()-1,X[:,1].max()+1,250))\n"
    "z=clf.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)\n"
    "plt.contourf(gx,gy,z,alpha=0.3,cmap='bwr')\nplt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',s=12)\nplt.title('支持向量分类器')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("14_Support_Vector_Machines_and_Flexible_Discriminants/14.2_The_Support_Vector_Classifier/14.2_The_Support_Vector_Classifier.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)