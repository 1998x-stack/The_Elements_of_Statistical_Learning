# 7.6_Multidimensional_Splines
"""多维样条: 二维径向基(RBF)对曲面的插值/平滑。"""
import warnings
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 400
    pts = rng.uniform(0, 1, (n, 2))
    z = np.sin(4 * pts[:, 0]) * np.cos(4 * pts[:, 1])
    # 自实现薄板/径向平滑: 用 RBF 插值
    rbf = RBFInterpolator(pts, z, smoothing=0.05)
    gx, gy = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
    zz = rbf(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(gx, gy, zz, cmap="viridis", alpha=0.8)
    ax.set_title("二维样条曲面 (RBF)")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.6_Multidimensional_Splines/7.6_surface.png", dpi=110)
    print("ok: 7.6")


if __name__ == "__main__":
    main()