from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


def get_silhouette_score(X: np.ndarray, k: int):
    """
    This function returns the best number of clusters based on the silhouette score.
    The silhouette analysis can be used to study the separation distance between the resulting clusters.

    Parameters
    ----------
    X : np.ndarray
        The features matrix.
    k : int
        The maximum number of clusters to test.

    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    """
    scores = []

    # For each number of clusters, calculate the silhouette score
    for n_clusters in range(2, k):
        clusterer = KMeans(init="k-means++", n_clusters=n_clusters, random_state=42)
        y = clusterer.fit_predict(X)

        scores.append((n_clusters, silhouette_score(X, y)))

    # And we return the number of clusters with the highest silhouette score
    return max(scores, key=lambda x: x[1])[0]
