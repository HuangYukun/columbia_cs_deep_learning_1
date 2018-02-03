import time
import numpy as np
from numpy.linalg import eigh

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################
    # data = X
    # data_mean = data.mean()
    # data_zero_mean = data.map(lambda obs: obs - data_mean)
    # data_zero_mean = list(map(lambda obs: obs - data_mean, data))
    # print(data_zero_mean)
    # cov = (data_zero_mean
    #           .map(lambda obs: np.outer(obs, obs))
    #           .reduce(lambda a, b: a + b)
    #       ) * 1. / data_zero_mean.count()
    # cov = np.cov(X)

    X1 = X - X.mean(axis=0)
    N = X1.shape[0]                # !!!
    fact = float(N - 1)
    cov = np.dot(X1.T, X1) / fact

    eig_vals, eig_vecs = eigh(cov)
    # print(eig_vecs.shape)

    inds = np.argsort(eig_vals)[::-1]
    # print(inds[:K].shape)
    # P = eig_vecs[:, inds[:K]]
    P = eig_vecs[inds[:K], :]
    T = 0

    # print(cov.shape)
    # print(X.shape)
    # print(P.shape)
    # print(P[0,0])
    # print(eig_vecs[:, inds[:K]].shape)
    # T = X.map(lambda obs: np.dot(obs, eig_vecs[:, inds[:K]]))
    # T = list(map(lambda obs: np.dot(obs, eig_vecs[:, inds[:K]]), X))
    # print(T.shape)

    # print(X.shape)


    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)
