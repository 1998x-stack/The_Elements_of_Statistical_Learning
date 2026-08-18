"""构建 12.4 深讲笔记本(指数损失 AdaBoost). 用法: python scripts/build_nb_124.py"""
import pathlib
import nbformat as nbf

md = [
    "# 12.4 指数损失与 AdaBoost（深讲）",
    "指数损失 $L(y,f)=e^{-y f(x)}$ 是 0-1 损失的光滑可微上界。",
    "## 1. 权重更新",
    "$$w_i \\leftarrow w_i e^{-\\alpha_m y_i G_m(x_i)}$$",
]
code = [
    "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.datasets import make_moons\n"
    "X,y=make_moons(n_samples=200,noise=0.2,random_state=0); y=np.where(y==1,1,-1)\n"
    "w=np.ones(len(X))/len(X); alphas=[]; ps=[]\n"
    "for m in range(40):\n"
    "    t=DecisionTreeClassifier(max_depth=1,random_state=m).fit(X,y,sample_weight=w)\n"
    "    p=t.predict(X); er=np.mean(w*(p!=y))/np.sum(w)\n"
    "    a=0.5*np.log((1-er)/(er+1e-12))\n"
    "    w=w*np.exp(-a*y*p); w/=w.sum(); alphas.append(a); ps.append(p)\n"
    "F=sum(a*p for a,p in zip(alphas,ps))\nprint('集成0-1误差', np.mean(np.sign(F)!=y))",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("12_Boosting_and_Additive_Trees/12.4_Exponential_Loss_and_AdaBoost/12.4_Exponential_Loss_and_AdaBoost.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)