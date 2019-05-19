from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

def create_data():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=0.5,random_state=3320)
    # =======================================
    # Complete the code here.
    # Plot the data points in a scatter plot.
    # Use color to represents the clusters.
    # =======================================
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    return [X, y]

def my_clustering(X, y, n_clusters):
    # =======================================
    # Complete the code here.
    # return scores like this: return [score, score, score, score]
    # =======================================
    y_pred = KMeans(n_clusters).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred.labels_)
    for i in range(n_clusters):
        plt.annotate(i, y_pred.cluster_centers_[i])
        plt.scatter(y_pred.cluster_centers_[i][0], y_pred.cluster_centers_[i][1], c = 'blue')
    plt.show()
    ARI = metrics.cluster.adjusted_rand_score(y, y_pred.labels_)
    MRI = metrics.cluster.mutual_info_score(y, y_pred.labels_)
    v_meansure = metrics.cluster.v_measure_score(y, y_pred.labels_)
    silhouette = metrics.cluster.silhouette_score(X, y_pred.labels_, metric='euclidean')
    return [ARI,MRI,v_meansure,silhouette]  # You won't need this line when you are done

def main():
    X, y = create_data()
    range_n_clusters = [2, 3, 4, 5, 6]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    # =======================================
    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters.
    # =======================================
    plt.plot(range_n_clusters, ari_score, label = 'Adjusted rand index')
    plt.plot(range_n_clusters, mri_score, label = 'Mutual Information based scores')
    plt.plot(range_n_clusters, v_measure_score, label = 'V-measure')
    plt.plot(range_n_clusters, silhouette_avg, label = 'Silhouette Coefficient')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()

