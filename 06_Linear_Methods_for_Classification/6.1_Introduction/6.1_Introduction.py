# 6.1_Introduction
"""第6章概述: 线性分类。"""
ROADMAP = [
    ("6.2", "指示矩阵的线性回归"),
    ("6.3", "线性判别分析 LDA"),
    ("6.4", "逻辑回归"),
    ("6.5", "分离超平面"),
]


def main() -> None:
    print("第6章: Linear Methods for Classification")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()