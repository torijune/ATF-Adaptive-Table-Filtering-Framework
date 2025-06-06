# K-means clustering

from langchain_core.runnables import RunnableLambda
import numpy as np
from sklearn.cluster import KMeans

def column_clustering_fn(state):
    score_dict = state["column_relevance_scores"]
    column_names = list(score_dict.keys())
    vectors = np.array([score_dict[c] for c in column_names])

    # print(f"[Clustering] Number of columns: {len(column_names)}")
    # print(f"[Clustering] Column names: {column_names}")
    # print(f"[Clustering] Vectors shape: {vectors.shape}")

    # Run k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(vectors)

    clustered = {col: int(cluster) for col, cluster in zip(column_names, labels)}

    print(f"[Clustering] Cluster labels: {clustered}")
    print(f"[Clustering] Cluster centers: {kmeans.cluster_centers_}")

    return {**state, "column_clusters": clustered, "cluster_centers": kmeans.cluster_centers_.tolist()}

column_clustering_node = RunnableLambda(column_clustering_fn)