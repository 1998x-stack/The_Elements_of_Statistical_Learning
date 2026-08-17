# 4.1_Introduction
"""第4章概述: 打印本章路线图。"""
ROADMAP = [
    ("4.2", "变量类型与术语"),
    ("4.3", "两类简单方法: 最小二乘与最近邻"),
    ("4.4", "统计决策理论"),
    ("4.9", "模型选择与偏差-方差权衡"),
]


def main() -> None:
    print("第4章: Overview of Supervised Learning")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()