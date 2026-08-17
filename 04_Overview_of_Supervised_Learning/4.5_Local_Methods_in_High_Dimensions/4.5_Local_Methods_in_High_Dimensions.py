# 4.5_Local_Methods_in_High_Dimensions
"""高维局部方法: 展示维数灾维中球的体积占比随维数下降。"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def ball_share(p, r=0.5):
    return math.pi ** (p / 2) / math.gamma(p / 2 + 1) * r ** p


def main() -> None:
    p = np.arange(1, 21)
    share = [ball_share(pp, 0.5) for pp in p]
    plt.plot(p, share, "-o")
    plt.xlabel("维数 p")
    plt.ylabel("半径为0.5的超球体积(单位立方体内)")
    plt.title("高维空间的\"维数灾维\"")
    plt.savefig("04_Overview_of_Supervised_Learning/4.5_Local_Methods_in_High_Dimensions/4.5_curse.png", dpi=110)
    print("ok: p=10 时半径0.5球体积占比 =", round(share[9], 4))


if __name__ == "__main__":
    main()