# 16.1_Introduction
"""第16章概述: 无监督学习。"""
ROADMAP = [
    ("16.2", "关联规则"),
    ("16.3-16.4", "聚类与自组织映射"),
    ("16.5-16.9", "降维: PCA/NMF/ICA/MDS/流形"),
    ("16.10", "PageRank"),
]


def main() -> None:
    print("第16章: Unsupervised Learning")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()