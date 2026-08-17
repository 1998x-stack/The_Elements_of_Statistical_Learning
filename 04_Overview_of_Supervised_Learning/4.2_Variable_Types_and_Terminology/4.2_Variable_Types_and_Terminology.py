# 4.2_Variable_Types_and_Terminology
"""变量类型: 数值 vs 类别, 类别用 one-hot 编码。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
    df = pd.DataFrame({"x1": [1.0, 2.5, 3.2], "x2": ["A", "B", "A"]})
    enc = OneHotEncoder()
    X = enc.fit_transform(df[["x2"]]).toarray()
    print("数值列:", df["x1"].to_numpy())
    print("类别 one-hot:\n", X)


if __name__ == "__main__":
    main()