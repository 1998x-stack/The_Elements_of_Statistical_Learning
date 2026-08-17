# 23_Index
"""示例: 用一个小型术语表演示关键词索引的构建。"""
GLOSSARY = {
    "Bias": "9.3",
    "Variance": "9.3",
    "Cross-Validation": "9.10",
    "Boosting": "12.4",
    "SVM": "14.2",
    "Random Forests": "17.2",
}


def main() -> None:
    for term, section in sorted(GLOSSARY.items()):
        print(f"{term: <18} -> 章节 {section}")


if __name__ == "__main__":
    main()