# 5.10_Bibliographic_Notes
"""第5章文献指引。"""
REFS = [
    "Hoerl & Kennard (1970): 岭回归",
    "Tibshirani (1996): Lasso",
    "Efron et al. (2004): LARS",
    "Wold (1975): PLS",
    "Stone & Jolliffe : 主成分回归",
]


def main() -> None:
    for r in REFS:
        print("-", r)


if __name__ == "__main__":
    main()