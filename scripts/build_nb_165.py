"""构建 16.5 深讲笔记本(PCA). 用法: python scripts/build_nb_165.py"""
import pathlib
import nbformat as nbf

md = [
    "# 16.5 主成分分析（深讲）",
    "PCA 找方差最大的方向并投影降维。",
    "## 1. 谱分解",
    "主成分为协方阵特征向量。",
    "## 2. 降维",
    "用前 K 主成分近似数据。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import load_iris\nfrom sklearn.decomposition import PCA\n"
    "X=load_iris().data\npca=PCA(n_components=2).fit(X)\nprint('解释方差比', np.round(pca.explained_variance_ratio_,3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['SimHei']\n"
    "Z=pca.transform(X)\nplt.scatter(Z[:,0],Z[:,1],c=load_iris().target,cmap='Set1',s=20)\nplt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA 2D')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("16_Unsupervised_Learning/16.5_Principal_Components,_Curves_and_Surfaces/16.5_Principal_Components,_Curves_and_Surfaces.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)