# 21_References
"""
Lecture: 21_References
Content: 21_References
内容: 提供 ESL 涉及的关键参考文献的快速检索辅助。
"""
KEY_REFS = [
    ("Hastie, Tibshirani & Friedman, 2009", "The Elements of Statistical Learning, 2nd ed."),
    ("Pearl, 2000", "Causality: Models, Reasoning, and Inference"),
    ("Breiman et al., 1984", "Classification and Regression Trees"),
    ("Bishop, 2006", "Pattern Recognition and Machine Learning"),
    ("Ripley, 1996", "Pattern Recognition and Neural Networks"),
]


def main() -> None:
    for author, title in KEY_REFS:
        print(f"- {author}: {title}")


if __name__ == "__main__":
    main()