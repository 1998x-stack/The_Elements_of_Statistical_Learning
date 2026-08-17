"""构建 4.5 深讲笔记本. 用法: python scripts/build_nb_45.py"""
import pathlib
import nbformat as nbf

md = [
    "# 4.5 高维空间的局部方法（深讲）",
    "本笔记本逐步复现\"维数灾维\"：随着维数增加，局部邻域迅速变空。",
    "## 1. 球与立方体的体积比",
    "在单位超立方体中放置半径 $r=0.5$ 的超球，其体积占比随维数指数下降。",
]
code = [
    "import math, numpy as np\n"
    "def share(p): return math.pi**(p/2)/math.gamma(p/2+1)*(0.5**p)\n"
    "print('p=10 占比 =', round(share(10), 5))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "p = np.arange(1,21)\nplt.plot(p, [share(x) for x in p], 'o')\n"
    "plt.xlabel('维数 p'); plt.ylabel('占比')\nplt.title('维数灾维')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("04_Overview_of_Supervised_Learning/4.5_Local_Methods_in_High_Dimensions/4.5_Local_Methods_in_High_Dimensions.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)