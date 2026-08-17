# 9.9_Vapnik_Chervonenkis_Dimension
"""VC 维度: 一维阈值分类器可打散(shatter)的点数。"""
import warnings
import numpy as np
from itertools import product
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    print("n  可实现标签数  总数(2^n)")
    for n in range(1, 6):
        total = 2 ** n
        real = 2 * n  # 阈值=前缀一致 0..0 1..1 或反向
        print(f"{n}   {real:9d}   {total}")

    print("\n干净校验(枚举全部标签并检查是否阈值可分):")
    for n in range(1, 4):
        total = 0
        count = 0
        for lab in product([0, 1], repeat=n):
            total += 1
            pos = any(all(v == 0 for v in lab[:i]) and all(v == 1 for v in lab[i:])
                      for i in range(n + 1))
            neg = any(all(v == 1 for v in lab[:i]) and all(v == 0 for v in lab[i:])
                      for i in range(n + 1))
            count += pos or neg
        print(f"n={n}: 可实现 {count}/{total}")
    print("结论: 一维阈值分类器在 n=2 时打散全部 2^2=4 种标签, 在 n=3 时"
          "仅 6<8 种 —— 故其 VC 维度 = 2。")


if __name__ == "__main__":
    main()