# 8.10_Bibliographic_Notes
"""第8章文献指引。"""
REFS = [
    "Silverman (1986): 密度估计(KDE)",
    "Cleveland (1979): loess 局部回归",
    "Fan & Gijbels (1996): 局部多项式回归",
    "Ripley (1996): 模式识别与混合模型",
]


def main() -> None:
    for r in REFS:
        print("-", r)


if __name__ == "__main__":
    main()