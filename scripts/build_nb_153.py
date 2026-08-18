"""构建 15.3 深讲笔记本(kNN). 用法: python scripts/build_nb_153.py"""
import pathlib
import nbformat as nbf

md = [
    "# 15.3 k-最近邻分类器（深讲）",
    "对测试点取最近 k 个训练样本做多数投票。",
    "## 1. 决策边界",
    "k 小则边界过拟合, k 大则过于光滑。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import make_blobs\nfrom sklearn.neighbors import KNeighborsClassifier\n"
    "X,y=make_blobs(n_samples=300,centers=3,cluster_std=1.5,random_state=0)\n"
    "knn=KNeighborsClassifier(n_neighbors=15).fit(X,y)\nprint('k=15 训练准确率', round(knn.score(X,y),3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['SimHei']\n"
    "gx,gy=np.meshgrid(np.linspace(X[:,0].min()-2,X[:,0].max()+2,200),np.linspace(X[:,1].min()-2,X[:,1].max()+2,200))\n"
    "z=knn.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)\n"
    "plt.contourf(gx,gy,z,alpha=0.25,cmap='Pastel1')\nplt.scatter(X[:,0],X[:,1],c=y,cmap='Set1',s=12)\nplt.title('kNN k=15 边界')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("15_Prototype_Methods_and_Nearest_Neighbors/15.3_k_Nearest_Neighbor_Classifiers/15.3_k_Nearest_Neighbor_Classifiers.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)