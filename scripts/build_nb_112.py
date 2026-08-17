"""构建 11.2 深讲笔记本(决策树). 用法: python scripts/build_nb_112.py"""
import pathlib
import nbformat as nbf

md = [
    "# 11.2 树方法（深讲）",
    "用递归二分构造决策树并可视化轴平行边界。",
    "## 1. 划分",
    "按特征阈值递归切分, 叶内给常数预测。",
    "## 2. 边界",
    "深度影响分段数目。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import make_circles\n"
    "from sklearn.tree import DecisionTreeClassifier\n"
    "X,y=make_circles(n_samples=300,noise=0.1,factor=0.5,random_state=0)\nclf=DecisionTreeClassifier(max_depth=4,random_state=0).fit(X,y)\nprint('深度',clf.get_depth(),'叶',clf.get_n_leaves())",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['SimHei']\n"
    "gx,gy=np.meshgrid(np.linspace(X[:,0].min()-0.2,X[:,0].max()+0.2,200),np.linspace(X[:,1].min()-0.2,X[:,1].max()+0.2,200))\n"
    "z=clf.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)\n"
    "plt.contourf(gx,gy,z,alpha=0.3,cmap='bwr')\nplt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',s=10)\nplt.title('决策树边界')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("11_Additive_Models,_Trees,_and_Related_Methods/11.2_Tree_Based_Methods/11.2_Tree_Based_Methods.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)