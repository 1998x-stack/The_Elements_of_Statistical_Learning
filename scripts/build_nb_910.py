"""构建 9.10 深讲笔记本(交叉验证). 用法: python scripts/build_nb_910.py"""
import pathlib
import nbformat as nbf

md = [
    "# 9.10 交叉验证（深讲）",
    "演示 K 折 CV 与 LOOCV 估计测试误差、选择复杂度。",
    "## 1. K 折",
    "将数据分成 K 份轮流留出。",
    "## 2. 复杂度曲线",
    "CV 误差随模型复杂度呈 U 型，用于选复杂度。",
]
code = [
    "import numpy as np\nfrom sklearn.preprocessing import PolynomialFeatures\n"
    "from sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import make_pipeline\n"
    "from sklearn.model_selection import cross_val_score\n"
    "rng=np.random.default_rng(0)\nx=np.sort(rng.uniform(-3,3,200)); y=np.sin(x)+0.2*rng.standard_normal(200)\n"
    "for d in (1,3,5,9):\n"
    "    m=make_pipeline(PolynomialFeatures(d),LinearRegression())\n"
    "    s=-cross_val_score(m,x[:,None],y,cv=5,scoring='neg_mean_squared_error').mean()\n"
    "    print('dg', d, 'CV-MSE', round(s,4))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','SimHei']\n"
    "ds=range(1,12)\nv=[-cross_val_score(make_pipeline(PolynomialFeatures(d),LinearRegression()),x[:,None],y,cv=5,scoring='neg_mean_squared_error').mean() for d in ds]\n"
    "plt.plot(list(ds),v,'-o')\nplt.xlabel('复杂次数'); plt.ylabel('5折CV误差'); plt.title('CV 选复杂度')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("09_Model_Assessment_and_Selection/9.10_Cross_Validation/9.10_Cross_Validation.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)