# 12.8_Example:_Spam_Data
"""垃圾邮件示例: 在一组高频特征上比较 boosting 与单模型。"""
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    # 用类似"宽特征"的低维代理演示 boosting 优于单树
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr, ytr)
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=0).fit(Xtr, ytr)
    print("单棵决策树准确率:", round(accuracy_score(yte, tree.predict(Xte)), 3))
    print("GBM 准确率:", round(accuracy_score(yte, gbm.predict(Xte)), 3))
    print("示例: (书中用垃圾邮件的组合特征; 这里用乳腺癌数据作演示)。")


if __name__ == "__main__":
    main()