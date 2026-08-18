# 14.1_Introduction
"""第14章概述: 支持向量机与灵活判别。"""
ROADMAP = [
    ("14.2", "支持向量分类器(最大间隔)"),
    ("14.3", "核方法与 SVM"),
    ("14.4-14.7", "从 LDA 到灵活/罚/混合判别"),
]


def main() -> None:
    print("第14章: Support Vector Machines and Flexible Discriminants")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()