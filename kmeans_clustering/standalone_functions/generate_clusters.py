import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from kmeans_clustering.standalone_functions.elbow_method import show_elbow_chart


def generate_clusters(
    df: pd.DataFrame,
    figure1: str,
    figure2: str,
    max_clusters: int = 7,
    show_elbow: bool = False,
) -> None:

    X_train, X_test, y_train, y_test = train_test_split(
        df[[figure1, figure2]],
        df[[figure2]],
        test_size=0.33,
        random_state=0,
    )
    X_train_norm = preprocessing.normalize(X_train)
    X_test_norm = preprocessing.normalize(X_test)

    if show_elbow:
        return show_elbow_chart(X_train_norm, max_clusters)

    model = KMeans(n_clusters=max_clusters, random_state=0, n_init="auto").fit(
        X_train_norm
    )

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=X_train, x="profits", y="marketValue", hue=model.labels_)

    plt.title("Profits vs Market Value Clusters")
    plt.xlabel("Profits")
    plt.ylabel("Market Value")

    plt.show()
