"""构建 13.3 深讲笔记本(神经网络). 用法: python scripts/build_nb_133.py"""
import pathlib
import nbformat as nbf

md = [
    "# 13.3 神经网络（深讲）",
    "单隐层网络 f_k = σ0(β0k + Σ w_jk g_j(x))，g_j 是隐激活。",
    "## 1. 决策面",
    "在二维合成数据上用 MLP 拟合并可视化。",
]
code = [
    "import numpy as np\nfrom sklearn.datasets import make_moons\n"
    "from sklearn.neural_network import MLPClassifier\nfrom sklearn.model_selection import train_test_split\n"
    "X,y=make_moons(n_samples=300,noise=0.15,random_state=0)\n"
    "Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0)\n"
    "mlp=MLPClassifier(hidden_layer_sizes=(10,),activation='tanh',max_iter=1000,random_state=0).fit(Xtr,ytr)\n"
    "print('测试准确率', round(mlp.score(Xte,yte),3))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['SimHei']\n"
    "gx,gy=np.meshgrid(np.linspace(X[:,0].min()-.2,X[:,0].max()+.2,250),np.linspace(X[:,1].min()-.2,X[:,1].max()+.2,250))\n"
    "z=mlp.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)\n"
    "plt.contourf(gx,gy,z,alpha=0.3,cmap='bwr')\nplt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',s=12)\nplt.title('单隐层 NN 决策面')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("13_Neural_Networks/13.3_Neural_Networks/13.3_Neural_Networks.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)