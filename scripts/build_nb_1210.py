"""构建 12.10 深讲笔记本(梯度提升). 用法: python scripts/build_nb_1210.py"""
import pathlib
import nbformat as nbf

md = [
    "# 12.10 梯度提升（深讲）",
    "在函数空间做梯度下降：每轮把负梯度（残差）拟合到小树上再累加。",
    "## 1. 负梯度",
    "平方损失下即残差 $y - f_{m-1}(x)$。",
    "## 2. 收缩",
    "以学习率 $\\eta$ 缩放每步。",
]
code = [
    "import numpy as np\nfrom sklearn.tree import DecisionTreeRegressor\n"
    "from sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error\n"
    "rng=np.random.default_rng(0)\nx=np.sort(rng.uniform(0,1,200)); y=np.sin(6*x)+0.2*rng.standard_normal(200)\n"
    "Xtr,Xte,ytr,yte=train_test_split(x[:,None],y,test_size=0.4,random_state=0)\n"
    "eta=0.1; Ftr=np.full_like(ytr,ytr.mean()); Fte=np.full_like(yte,ytr.mean())\n"
    "for m in range(150):\n"
    "    t=DecisionTreeRegressor(max_depth=3,random_state=m).fit(Xtr,ytr-Ftr)\n"
    "    Ftr+=eta*t.predict(Xtr); Fte+=eta*t.predict(Xte)\n"
    "print('GBM(150,深度3,eta=0.1) 测试MSE', round(mean_squared_error(yte,Fte),4))",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("12_Boosting_and_Additive_Trees/12.10_Numerical_Optimization_via_Gradient_Boosting/12.10_Numerical_Optimization_via_Gradient_Boosting.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)