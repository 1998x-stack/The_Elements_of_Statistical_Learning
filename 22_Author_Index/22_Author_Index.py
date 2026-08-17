# 22_Author_Index
"""生成一个按字母排序的作者索引示例。"""
AUTHORS = ["Tibshirani", "Hastie", "Friedman", "Breiman", "Vapnik", "Pearson", "Bishop"]


def main() -> None:
    for name in sorted(AUTHORS, key=str.lower):
        print(name)


if __name__ == "__main__":
    main()