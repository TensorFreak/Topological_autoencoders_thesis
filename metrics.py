import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform

import torch

from torch_topological.nn import SignatureLoss
from torch_topological.nn import VietorisRipsComplex
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler



def estimate_kl_divergence_kde(X_input, X_embed, bandwidth_input=None, bandwidth_embed=None):

    assert len(X_input) == len(X_embed), "Point clouds must have the same number of points."

    X_input = normalize_points_cloud(X_input)
    X_embed = normalize_points_cloud(X_embed)
    #transform for same number of features
    if X_input.shape[1] != X_embed.shape[1]:
        min_dim = min(X_input.shape[1], X_embed.shape[1])
        pca = PCA(n_components=min_dim)
        X_input = pca.fit_transform(X_input)
        X_embed = pca.fit_transform(X_embed)
    
    # Fit KDE models
    kde_input = gaussian_kde(X_input.T, bw_method=bandwidth_input)
    kde_embed = gaussian_kde(X_embed.T, bw_method=bandwidth_embed)
    
    # Compute log densities for X_input under both KDEs
    log_p_input = kde_input.logpdf(X_input.T)  # log p_input(x)
    log_q_input = kde_embed.logpdf(X_input.T)  # log p_embed(x)
    
    # KL divergence: D_KL(P_input || P_embed) = E_{x ~ P_input}[log p_input(x) - log p_embed(x)]
    kl_divergence = np.mean(log_p_input - log_q_input)
    
    return kl_divergence

    
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


def compute_klsigma(X_input, X_latent, sigma=1, num_grid_points=1000):
    """
    Compute KL divergence between KDEs of input and latent representations.
    
    Parameters:
        X_input (np.ndarray): Original input representations, shape (n_samples, N).
        X_latent (np.ndarray): Latent space representations, shape (n_samples, M).
        sigma (float): Bandwidth for Gaussian kernel (length scale σ).
        num_grid_points (int): Number of points in the shared evaluation grid.

    Returns:
        float: KL divergence KL(p_input || p_latent)
    """
    scaler = StandardScaler()
    X_input = scaler.fit_transform(X_input)
    X_latent = scaler.fit_transform(X_latent)
    
    pca = PCA(n_components=1) # try min_dim later
    X_input = pca.fit_transform(X_input)
    X_latent = pca.fit_transform(X_latent)
    
    # Fit KDEs
    kde_input = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X_input)
    kde_latent = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X_latent)

    # Define shared grid for evaluation
    x_min = min(X_input.min(), X_latent.min())
    x_max = max(X_input.max(), X_latent.max())
    X_eval = np.linspace(x_min, x_max, num_grid_points).reshape(-1, 1)

    # Evaluate log densities
    log_p_input = kde_input.score_samples(X_eval)
    log_p_latent = kde_latent.score_samples(X_eval)

    # Convert to probability distributions
    p_input = np.exp(log_p_input)
    p_latent = np.exp(log_p_latent)

    # Normalize to ensure they sum to 1 (discrete approximation)
    p_input /= np.sum(p_input)
    p_latent /= np.sum(p_latent)

    # Compute KL divergence: KL(p_input || p_latent)
    kl_div = entropy(p_input, p_latent)
    
    return kl_div