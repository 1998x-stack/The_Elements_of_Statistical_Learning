"""构建 16.3 深讲笔记本(聚类). 用法: python scripts/build_nb_163.py"""
import pathlib
import nbformat as nbf

md = [
    "# 16.3 聚类分析（深讲）",
    "比较 k-means 与层次聚类并展示聚类中心。",
    "## 1. k-means",
    "$$\\min\\sum_k\\sum_{i\\in C_k}\\|x_i-\\mu_k\\|^2$$",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import make_blobs\nfrom sklearn.cluster import KMeans\n"
    "X,y=make_blobs(n_samples=300,centers=4,cluster_std=0.9,random_state=0)\n"
    "km=KMeans(n_clusters=4,n_init=10,random_state=0).fit(X)\nprint('惯性', round(km.inertia_,2))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['SimHei']\n"
    "plt.scatter(X[:,0],X[:,1],c=km.labels_,cmap='viridis',s=12)\nplt.plot(km.cluster_centers_[:,0],km.cluster_centers_[:,1],'rx',ms=15)\nplt.title('k-means')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("16_Unsupervised_Learning/16.3_Cluster_Analysis/16.3_Cluster_Analysis.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)