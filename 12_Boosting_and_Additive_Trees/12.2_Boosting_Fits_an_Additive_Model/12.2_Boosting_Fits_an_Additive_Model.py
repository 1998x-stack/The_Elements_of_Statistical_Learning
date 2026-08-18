# 12.2_Boosting_Fits_an_Additive_Model
"""Boosting 是加性模型: 逐步累加弱学习器 f(x)=Σ α_m G_m(x)。"""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    # 用简单 step 弱学习器演示累加成复杂函数
    x = np.linspace(0, 1, 100)
    weak = [np.where(x > 0.2 * i, 1.0, 0.0) for i in range(1, 6)]
    F = np.zeros_like(x)
    print("加性累加:")
    for m, G in enumerate(weak, 1):
        F += G
        print(f"  m={m}: f_m 幅度={G.sum():.0f}, 累加 F 均值至今={F.mean():.2f}")
    print("结论: boosting 是 f(x)=Σ αₘ Gₘ(x) 的加性过程, 每步只加一个小基学习器。")


if __name__ == "__main__":
    main()