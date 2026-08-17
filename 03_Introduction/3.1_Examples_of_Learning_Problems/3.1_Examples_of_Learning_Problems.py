# 3.1_Examples_of_Learning_Problems
"""示例: 展示监督回归与无监督聚类的两类学习问题。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)


def main() -> None:
    x = np.sort(rng.uniform(0, 4, 200))
    y = np.sin(1.5 * x) + 0.2 * rng.standard_normal(200)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.5))
    a.scatter(x, y, s=8)
    a.set_title("监督学习: 由 X 预测实值 Y (回归)")
    pts = np.vstack([rng.multivariate_normal([0, 0], np.eye(2), 75),
                     rng.multivariate_normal([2, 2], 0.7 * np.eye(2), 75)])
    b.scatter(pts[:, 0], pts[:, 1], s=8, c=pts[:, 0] > 1)
    b.set_title("无监督学习: 数据中的簇结构")
    fig.tight_layout()
    fig.savefig("03_Introduction/3.1_Examples_of_Learning_Problems/3.1_Examples_of_Learning_Problems_图.png", dpi=110)
    print("ok: 3.1 已生成示例图")


if __name__ == "__main__":
    main()