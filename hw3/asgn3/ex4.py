from __future__ import print_function

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import misc
from struct import unpack

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from numpy import linalg

def plot_mean_image(X, log):
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(meanrow,(28,28)), cmap=plt.cm.binary)
    plt.title('Mean image of ' + log)
    plt.show()

def get_labeled_data(imagefile, labelfile):
    """
    Read input-vector (image) and target class (label, 0-9) and return it as list of tuples.
    Adapted from: https://martin-thoma.com/classify-mnist-with-pybrain/
    """
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Read the binary data
    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    X = np.zeros((N, rows * cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for id in range(rows * cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            X[i][id] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (X, y)


def my_clustering(X, y, n_clusters, pca):
    # =======================================
    # Complete the code here.
    # return scores like this: return [score, score, score, score]
    # =======================================
    y_pred = KMeans(n_clusters).fit(X)
    # plot_mean_image(y_pred.cluster_centers_, "center of cluster")
    # print(y_pred.cluster_centers_.shape)
    for i in range(n_clusters):
        # print(y_pred.cluster_centers_[i].shape)
        plot_mean_image(pca.inverse_transform(y_pred.cluster_centers_[i]).reshape(1, 784), "center of cluster")
    ARI = metrics.cluster.adjusted_rand_score(y, y_pred.labels_)
    MRI = metrics.cluster.mutual_info_score(y, y_pred.labels_)
    v_meansure = metrics.cluster.v_measure_score(y, y_pred.labels_)
    silhouette = metrics.cluster.silhouette_score(X, y_pred.labels_, metric='euclidean')
    return [ARI,MRI,v_meansure,silhouette]  # You won't need this line when you are done

def main():
    # Load the dataset
    fname_img = 't10k-images.idx3-ubyte'
    fname_lbl = 't10k-labels.idx1-ubyte'
    [X, y]=get_labeled_data(fname_img, fname_lbl)

    # Plot the mean image
    plot_mean_image(X, 'all images')

    # =======================================
    # Complete the code here.
    # Use PCA to reduce the dimension here.
    # You may want to use the following codes. Feel free to write in your own way.
    # - pca = PCA(n_components=...)
    # - pca.fit(X)
    # - print('We need', pca.n_components_, 'dimensions to preserve 0.95 POV')
    # =======================================

    # apply PCA, and get new data
    covariance = np.cov(X.T)
    D, V = linalg.eig(covariance)
    -np.sort(-D)
    deno = D.sum()
    k=0
    for k in range(2, 784):
        if D[0:k].sum()/deno >= 0.95:
            break
    # print("k is", k)
    pca = PCA(k)
    X = pca.fit_transform(X)
    # print(new_X.shape)

    # Clustering
    range_n_clusters = [8, 9, 10, 11, 12]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters, pca)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    # =======================================
    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters.
    # ======================================= 
    plt.figure(figsize=(5.0 * 2, 3.0 * 2))
    plt.subplot(2, 2, 1)
    plt.plot(range_n_clusters, ari_score)
    plt.title('Adjusted rand index')
    plt.subplot(2, 2, 2)
    plt.plot(range_n_clusters, mri_score)
    plt.title('Mutual Information based scores')
    plt.subplot(2, 2, 3)
    plt.plot(range_n_clusters, v_measure_score)
    plt.title('V-measure')
    plt.subplot(2, 2, 4)
    plt.plot(range_n_clusters, silhouette_avg)
    plt.title('Silhouette Coefficient')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
