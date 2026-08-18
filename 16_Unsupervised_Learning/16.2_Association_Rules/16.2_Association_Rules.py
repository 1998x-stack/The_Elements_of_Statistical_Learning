# 16.2_Association_Rules
"""关联规则: 用 apriori 思想找频繁项集并算提升度。"""
import warnings
import numpy as np
from itertools import combinations
from collections import Counter
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n, items = 1000, ["milk", "bread", "butter", "beer", "eggs"]
    # 生成购物篮
    baskets = []
    for _ in range(n):
        s = {it for it in items if rng.random() < 0.3}
        if "milk" in s and rng.random() < 0.6:
            s.add("bread")     # milk→bread 有较强关联
        baskets.append(s)
    total = len(baskets)
    sup = {it: sum(it in b for b in baskets) / total for it in items}
    pairs = list(combinations(items, 2))
    print("支持度最高的项对:")
    ranked = []
    for a, b in pairs:
        ab = sum(a in bc and b in bc for bc in baskets) / total
        conf = ab / sup[a]
        ranked.append((a, b, round(ab, 3), round(conf, 2)))
    ranked.sort(key=lambda t: -t[2])
    for a, b, ab, conf in ranked[:3]:
        print(f"  {a}->{b}: 支持={ab} 置信={conf}")
    print("提升度>1 表示正关联。")


if __name__ == "__main__":
    main()