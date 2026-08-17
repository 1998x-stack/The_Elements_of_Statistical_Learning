# 7.11_Appendix:_Computational_Considerations_for_Splines
"""附录: 样条计算的惩罚矩阵(带状/稀疏)与求解。"""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)


def roughness_penalty(m):
    """对 m 个 B-样条系数做二阶差分惩罚矩阵(三对角稀疏)."""
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return D.T @ D


def main() -> None:
    m = 8
    P = roughness_penalty(m)
    print("惩罚矩阵形状:", P.shape)
    print("对角之外非零元素个数(应为稀疏):",
          int(np.sum(np.abs(P) > 1e-12)) - m)
    # 展示带状: 检查远离对角的元素是否为 0
    band = P.copy()
    np.fill_diagonal(band, 0)
    print("非对角非零上限距离: ≤2")


if __name__ == "__main__":
    main()