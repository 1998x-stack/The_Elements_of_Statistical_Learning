# 10.1_Introduction
"""第10章概述: 模型推断与平均化。"""
ROADMAP = [
    ("10.2-10.4", "自举、MLE 与贝叶斯推断"),
    ("10.5-10.6", "EM 与 MCMC"),
    ("10.7-10.9", "Bagging、模型平均/Stacking、Bumping"),
]


def main() -> None:
    print("第10章: Model Inference and Averaging")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()