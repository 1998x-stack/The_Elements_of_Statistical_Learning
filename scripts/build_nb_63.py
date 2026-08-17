"""构建 6.3 深讲笔记本(LDA). 用法: python scripts/build_nb_63.py"""
import pathlib
import nbformat as nbf

md = [
    "# 6.3 线性判别分析 (LDA)（深讲）",
    "推导判别函数并在鸢尾花数据上可视化决策区域。",
    "## 1. 判别函数",
    "$$\\delta_k(x)=x^T\\Sigma^{-1}\\mu_k-\\tfrac12\\mu_k^T\\Sigma^{-1}\\mu_k+\\log\\pi_k$$",
    "## 2. 实例",
    "使用鸢尾花前两个特征。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import load_iris\n"
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
    "iris=load_iris(); X=iris.data[:,:2]; y=iris.target\n"
    "lda=LinearDiscriminantAnalysis().fit(X,y)\nprint('训练准确率', round(lda.score(X,y),3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "plt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "gx,gy=np.meshgrid(np.linspace(X[:,0].min()-1,X[:,0].max()+1,300),np.linspace(X[:,1].min()-1,X[:,1].max()+1,300))\n"
    "Z=lda.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)\n"
    "plt.contourf(gx,gy,Z,alpha=0.3,cmap='Pastel1')\nplt.scatter(X[:,0],X[:,1],c=y,cmap='Set1',s=15)\nplt.title('LDA 决策区域')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("06_Linear_Methods_for_Classification/6.3_Linear_Discriminant_Analysis/6.3_Linear_Discriminant_Analysis.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)