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
    """
    This function is used to easily generate clusters for a given dataset and the fields to be used for clustering.

    This way, you can observe the clusters formed by the KMeans algorithm between figure1 and figure2.
    You can  also show the elbow chart to determine the optimal number of clusters (you need to properly import the show_elbow_chart function).

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be used for clustering. The data need to be cleaned and preprocessed before passing it to this function.

    figure1 : str
        The first field to be used for clustering. This field should be present in the dataset.

    figure2 : str
        The second field to be used for clustering. This field should be present in the dataset.

    max_clusters : int
        The maximum number of clusters to be formed. The default value is 7.

    show_elbow : bool
        A flag to show the elbow chart or not. The default value is False. You can show it to decide the optimal number of clusters and then run the
        function again with the optimal number of clusters.


    """

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
