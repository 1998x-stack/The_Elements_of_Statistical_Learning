"""构建 12.1 深讲笔记本(AdaBoost). 用法: python scripts/build_nb_121.py"""
import pathlib
import nbformat as nbf

md = [
    "# 12.1 Boosting（深讲）",
    "手写 AdaBoost：每轮对弱学习器重新加权，组合成强学习器。",
    "## 1. 权重更新",
    "$$w_i \\leftarrow w_i e^{\\alpha_m\\mathbf 1_{[G_m(x_i)\\ne y_i]}}$$",
]
code = [
    "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.datasets import make_moons\n"
    "X,y=make_moons(n_samples=200,noise=0.2,random_state=0); y=np.where(y==1,1,-1)\n"
    "w=np.ones(len(X))/len(X); al=np.array([]); ts=[]\n"
    "for m in range(30):\n"
    "    t=DecisionTreeClassifier(max_depth=1,random_state=m).fit(X,y,sample_weight=w)\n"
    "    pred=t.predict(X); err=np.sum(w*(pred!=y))/w.sum()\n"
    "    a=0.5*np.log((1-err)/(err+1e-12))\n"
    "    w=w*np.exp(-a*y*pred); w=w/w.sum()\n"
    "    ts.append(t); alphas=np.append(alphas,a)\n"
    "F=sum(ai*t.predict(X) for ai,t in zip(alphas,ts))\n"
    "print('训练误差', np.mean(np.sign(F)!=y))",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("12_Boosting_and_Additive_Trees/12.1_Boosting_Methods/12.1_Boosting_Methods.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)