import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def show_elbow_chart(X_train: np.ndarray, max_clusters: int) -> None:
    """
    Display the chart of the elbow method.

    Parameters
    ----------
    max_clusters : int
        The maximum number of clusters to iterate through. Feel free to change this value to see the elbow chart for a different range of clusters.

    X_train : np.ndarray
        The training data. These are the features on which the model will be trained.

    https://en.wikipedia.org/wiki/Elbow_method_(clustering)
    """
    fits = []
    scores = []

    K = range(2, max_clusters + 1)

    for k in K:
        # Train the model for the current value of k on training data
        model = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_train)

        # Append the model to fits
        fits.append(model)

        # Calculate the silhouette score
        score = silhouette_score(X_train, model.labels_, metric="euclidean")

        # Append the silhouette score to scores
        scores.append(score)

    # Plotting the elbow chart
    plt.plot(K, scores, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Elbow Method For Optimal k")
    plt.show()
