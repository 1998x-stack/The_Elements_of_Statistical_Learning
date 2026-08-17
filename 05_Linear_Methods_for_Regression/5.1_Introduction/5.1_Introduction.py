# 5.1_Introduction
"""第5章概述: 线性方法回归。"""
ROADMAP = [
    ("5.2", "线性回归与最小二乘"),
    ("5.3-5.4", "子集选择与收缩(岭/Lasso)"),
    ("5.5", "派生输入方向(PCR/PLS)"),
    ("5.9", "计算问题"),
]


def main() -> None:
    print("第5章: Linear Methods for Regression")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()