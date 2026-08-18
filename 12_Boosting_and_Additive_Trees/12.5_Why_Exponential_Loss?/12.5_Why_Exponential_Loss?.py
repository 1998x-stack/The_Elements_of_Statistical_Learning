# 12.5_Why_Exponential_Loss?
"""为什么用指数损失: 与 0-1 损失的对比及可微上界。"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    z = np.linspace(-3, 3, 300)          # margin = y*f(x)
    zero_one = (z < 0).astype(float)
    hinge = np.maximum(0, 1 - z)          # hinge (SVM)
    exp = np.exp(-z)
    plt.plot(z, zero_one, label="0-1 损失")
    plt.plot(z, exp, label="指数损失 e^{-z}")
    plt.plot(z, hinge, label="Hinge log(1+... )")
    plt.legend()
    plt.title("指数损失是 0-1 的光滑上界")
    plt.savefig("12_Boosting_and_Additive_Trees/12.5_Why_Exponential_Loss?/12.5_loss.png", dpi=110)
    print("ok: 12.5 指数损失 e^-z 可微、凸, 是 0-1 的上界; 对分类友好但对噪声敏感。")


if __name__ == "__main__":
    main()