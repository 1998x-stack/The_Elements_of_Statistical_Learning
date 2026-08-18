# 16.3_Cluster_Analysis
"""聚类分析: k-means 与层次聚类的对比。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=0)
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    ag = AgglomerativeClustering(n_clusters=4).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=12)
    plt.title("K-means 聚类 (K=4)")
    plt.savefig("16_Unsupervised_Learning/16.3_Cluster_Analysis/16.3_kmeans.png", dpi=110)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=ag.labels_, cmap="viridis", s=12)
    plt.title("层次平均聚类")
    plt.savefig("16_Unsupervised_Learning/16.3_Cluster_Analysis/16.3_agglo.png", dpi=110)
    print("k-means 与层次聚类完成; 二者差异见图中。")


if __name__ == "__main__":
    main()