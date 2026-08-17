# 9.1_Introduction
"""第9章概述: 模型评估与选择。"""
ROADMAP = [
    ("9.2-9.3", "偏差、方差与模型复杂度"),
    ("9.4-9.6", "训练误差的乐观性、样本内误差、有效参数数"),
    ("9.7-9.9", "BIC、MDL 与 VC 维度"),
    ("9.10-9.11", "交叉验证与自举"),
]


def main() -> None:
    print("第9章: Model Assessment and Selection")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()