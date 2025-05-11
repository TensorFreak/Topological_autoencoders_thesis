import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import MinMaxScaler

from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

import torch

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex


def normalize_points_cloud(X):
    X_norm = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)  
    return X_norm

def knn_neighborhood_preservation(X_orig, X_latent, k=5):
    """
    Computes the fraction of preserved k-nearest neighbors between original and latent space.
    """
    X_orig = normalize_points_cloud(X_orig)
    X_latent = normalize_points_cloud(X_latent)

    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_orig)
    distances_orig, indices_orig = nbrs_orig.kneighbors(X_orig)
    
    nbrs_latent = NearestNeighbors(n_neighbors=k+1).fit(X_latent)
    distances_latent, indices_latent = nbrs_latent.kneighbors(X_latent)
    
    preservation = 0.0
    for i in range(len(X_orig)):
        common_neighbors = len(set(indices_orig[i, 1:k+1]) & set(indices_latent[i, 1:k+1]))
        preservation += common_neighbors / k
    return preservation / len(X_orig)

def spearman_distance_correlation(X_orig, X_latent):
    """
    Computes Spearman's rank correlation between pairwise distances in original and latent space.
    """
    X_orig = normalize_points_cloud(X_orig)
    X_latent = normalize_points_cloud(X_latent)
    dist_orig = pdist(X_orig)
    dist_latent = pdist(X_latent)
    corr, _ = spearmanr(dist_orig, dist_latent)
    return corr

def compute_trustworthiness(X_orig, X_latent, k=5):
    """
    Computes trustworthiness (preservation of local structure).
    """
    X_orig = normalize_points_cloud(X_orig)
    X_latent = normalize_points_cloud(X_latent)
    return trustworthiness(X_orig, X_latent, n_neighbors=k)


def procrustes_error(X_orig, X_latent):
    """
    Computes Procrustes error after aligning latent space to original space.
    """
    X_orig = normalize_points_cloud(X_orig)
    X_latent = normalize_points_cloud(X_latent)
    R, scale = orthogonal_procrustes(X_latent, X_orig)
    aligned_latent = X_latent @ R
    error = np.mean(np.sqrt(np.sum((aligned_latent - X_orig)**2, axis=1)))
    return error

def mean_relative_rank_error(X_orig, X_latent, k=5):
    """
    Computes Mean Relative Rank Error (MRRE) between original and latent space.
    """
    X_orig = normalize_points_cloud(X_orig)
    X_latent = normalize_points_cloud(X_latent)
    n = len(X_orig)
    nbrs_orig = NearestNeighbors(n_neighbors=n).fit(X_orig)
    distances_orig, indices_orig = nbrs_orig.kneighbors(X_orig)
    
    nbrs_latent = NearestNeighbors(n_neighbors=n).fit(X_latent)
    distances_latent, indices_latent = nbrs_latent.kneighbors(X_latent)
    
    mrre = 0.0
    for i in range(n):
        for j in range(1, k+1):  # Exclude self (rank 0)
            neighbor = indices_orig[i, j]
            rank_orig = j
            rank_latent = np.where(indices_latent[i] == neighbor)[0][0]
            mrre += abs(rank_orig - rank_latent) / rank_orig
    return mrre / (n * k)

def calculate_top_loss(X_orig, X_latent, max_dim, loss_dims):
    X_orig = torch.tensor(X_orig)
    X_latent = torch.tensor(X_latent)

    vr_complex = VietorisRipsComplex(dim=max_dim)
    loss = SignatureLoss(dimensions=loss_dims)


    pi_orig = vr_complex(X_orig)
    pi_latent = vr_complex(X_latent)

    top_loss = loss([X_orig, pi_orig], [X_latent, pi_latent])
    return top_loss.item()