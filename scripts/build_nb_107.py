"""构建 10.7 深讲笔记本(bagging). 用法: python scripts/build_nb_107.py"""
import pathlib
import nbformat as nbf

md = [
    "# 10.7 Bagging（深讲）",
    "对重采样后的决策树做预测平均以降低方差。",
    "## 1. 单棵树 vs Bagging",
    "比较测试 MSE。",
    "## 2. 收敛",
    "随 B 增大看 bagging 误差趋稳。",
]
code = [
    "import numpy as np\nfrom sklearn.tree import DecisionTreeRegressor\n"
    "from sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error\n"
    "rng=np.random.default_rng(0)\nx=np.sort(rng.uniform(0,1,200)); y=np.sin(6*x)+0.2*rng.standard_normal(200)\n"
    "Xtr,Xte,ytr,yte=train_test_split(x[:,None],y,test_size=0.4,random_state=0)\n"
    "tree=DecisionTreeRegressor(max_depth=8,random_state=0).fit(Xtr,ytr)\n"
    "preds=np.zeros((100,len(Xte)))\n"
    "for b in range(100):\n"
    "    idx=rng.integers(0,len(Xtr),len(Xtr))\n"
    "    preds[b]=DecisionTreeRegressor(max_depth=8,random_state=b).fit(Xtr[idx],ytr[idx]).predict(Xte)\n"
    "print('单棵 MSE', round(mean_squared_error(yte,tree.predict(Xte)),4))\n"
    "print('bagging MSE', round(mean_squared_error(yte,preds.mean(0)),4))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['SimHei']\n"
    "steps=np.arange(1,101)\nerr=[mean_squared_error(yte,preds[:s].mean(0)) for s in steps]\n"
    "plt.plot(steps,err)\nplt.xlabel('B'); plt.ylabel('bagging测试MSE')\nplt.title('Bagging 随 B 收敛')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("10_Model_Inference_and_Averaging/10.7_Bagging/10.7_Bagging.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)