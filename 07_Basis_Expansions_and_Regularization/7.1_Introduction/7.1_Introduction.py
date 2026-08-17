# 7.1_Introduction
"""第7章概述: 基展开与正则化。"""
ROADMAP = [
    ("7.2-7.4", "样条与平滑样条"),
    ("7.5-7.6", "非参数逻辑回归与多维样条"),
    ("7.7", "正则化与再生核(RKHS)"),
    ("7.8", "小波平滑"),
]


def main() -> None:
    print("第7章: Basis Expansions and Regularization")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()