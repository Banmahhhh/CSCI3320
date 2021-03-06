import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy import linalg


def create_data(x1, x2, x3):
    x4 = -4.0 * x1
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    x7 = np.multiply(x2, x2)
    x8 = -1 * x3 / 10
    x9 = 2.0 * x3 + 2.0
    X = np.hstack((x1, x2, x3, x4, x5, x6, x7, x8, x9))
    return X

def pca(X):
    '''
    # PCA step by step
    #   1. normalize matrix X
    #   2. compute the covariance matrix of the normalized matrix X
    #   3. do the eigenvalue decomposition on the covariance matrix
    # If you do not remember Eigenvalue Decomposition, please review the linear
    # algebra
    # In this assignment, we use the ``unbiased estimator'' of covariance. You
    # can refer to this website for more information
    # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    # Actually, Singular Value Decomposition (SVD) is another way to do the
    # PCA, if you are interested, you can google SVD.
    # YOUR CODE HERE!
    '''
    ####################################################################
    # here V is the matrix containing all the eigenvectors, D is the
    # column vector containing all the corresponding eigenvalues.
    normalizedX = X-X.mean(0)
    covariance = np.cov(normalizedX.T)
    # print(covariance)
    D, V = linalg.eig(covariance)
    V = V.T
    order = np.argsort(D)[::-1]
    D = D[order]
    V = V[order]
    return [V, D]


def main():
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape) # samples from normal distribution
    x2 = np.random.exponential(10.0, shape) # samples from exponential distribution
    x3 = np.random.uniform(-100, 100, shape) # uniformly sampled data points
    X = create_data(x1, x2, x3)
    ####################################################################
    # Use the definition in the lecture notes,
    #   1. perform PCA on matrix X
    #   2. plot the eigenvalues against the order of eigenvalues,
    #   3. plot POV v.s. the order of eigenvalues
    # YOUR CODE HERE!

    ####################################################################
    V, D = pca(X)
    print("V is")
    print(V)
    print("D is")
    print(D)
    deno = D.sum()
    pov = np.array([D[0]/deno, D[0:2].sum()/deno, D[0:3].sum()/deno, D[0:4].sum()/deno, D[0:5].sum()/deno, D[0:6].sum()/deno, D[0:7].sum()/deno, D[0:8].sum()/deno, D.sum()/deno])
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(range(1, 10), D)
    plt.title('Eigenvalues')
    # plt.show()
    plt.subplot(2, 1, 2)
    plt.plot(range(1, 10), pov)
    plt.title("PoV")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

