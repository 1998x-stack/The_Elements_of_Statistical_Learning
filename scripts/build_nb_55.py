"""构建 5.5 深讲笔记本(PCR/PLS). 用法: python scripts/build_nb_55.py"""
import pathlib
import nbformat as nbf

md = [
    "# 5.5 派生输入方向: PCR 与 PLS（深讲）",
    "展示如何用无监督(PCA)或有监督(PLS)的方式得到低维方向后再回归。",
    "## 1. PCR",
    "主成分方向来自 X。",
    "## 2. 与 OLS 对比",
    "在强相关问题下，PCR/PLS 以偏差换方差。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import make_regression\n"
    "from sklearn.decomposition import PCA\nfrom sklearn.linear_model import LinearRegression\n"
    "from sklearn.cross_decomposition import PLSRegression\nfrom sklearn.model_selection import train_test_split\n"
    "from sklearn.metrics import mean_squared_error\n"
    "X,y=make_regression(n_samples=120,n_features=10,noise=0.6,random_state=1)\n"
    "Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0)",
    "pca=PCA(n_components=4); Xp=pca.fit_transform(Xtr)\n"
    "m=LinearRegression().fit(Xp,ytr)\n"
    "print('PCR(4) 测试MSE =', round(mean_squared_error(yte, m.predict(pca.transform(Xte))),2))",
    "pls=PLSRegression(n_components=4).fit(Xtr,ytr)\n"
    "print('PLS(4) 测试MSE =', round(mean_squared_error(yte, pls.predict(Xte)),2))",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("05_Linear_Methods_for_Regression/5.5_Methods_Using_Derived_Input_Directions/5.5_Methods_Using_Derived_Input_Directions.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)