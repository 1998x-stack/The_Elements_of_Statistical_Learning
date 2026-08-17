# 10.4_Relationship_Between_the_Bootstrap_and_Bayesian_Inference
"""自举与贝叶斯的关系: 非参数自举作为粗略后验近似。"""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = rng.gamma(2.0, 1.0, 100)  # 偏态数据, 均值统计量
    B = 2000
    boot_means = []
    for _ in range(B):
        boot_means.append(x[rng.integers(0, len(x), len(x))].mean())
    boot_means = np.array(boot_means)
    print(f"样本均值 = {x.mean():.3f}")
    print(f"自举均值 ±1.96se = {(x.mean() - 1.96*boot_means.std()):.3f} .. {(x.mean() + 1.96*boot_means.std()):.3f}")
    print("自举分布接近参数 bootstrap 之下均值的抽样分布——一种非参数贝叶斯式的后验近似。")


if __name__ == "__main__":
    main()