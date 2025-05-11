# modified Signature Loss with dimentions weighting

import torch
from typing import Union, Tuple, List

class SignatureLoss(torch.nn.Module):
    """Topological signature loss with dimension-wise weighting.

    Parameters
    ----------
    p : float
        Exponent for the p-norm distance calculation (default=2, Euclidean).
    normalise : bool
        If True, normalizes distances by the maximum distance in each point cloud.
    dimensions : Union[int, Tuple[int, ...]]
        Dimensions to consider in the topological signature (e.g., (0,1) for connected components and holes).
    weights : Union[None, List[float], torch.Tensor]
        Weights for each dimension in `dimensions`. If None, uniform weighting is applied.
    """
    def __init__(
        self,
        p: float = 2,
        normalise: bool = True,
        dimensions: Union[int, Tuple[int, ...]] = 0,
        weights: Union[None, List[float], torch.Tensor] = None,
    ):
        super().__init__()
        self.p = p
        self.normalise = normalise
        self.dimensions = [dimensions] if isinstance(dimensions, int) else list(dimensions)
        
        # Initialize weights (uniform if None)
        if weights is None:
            self.weights = torch.ones(len(self.dimensions))
        else:
            self.weights = torch.tensor(list(weights), dtype=torch.float32)# if isinstance(weights, list) else weights
        
        # Ensure weights sum to 1 (optional, for interpretability)
        self.weights = self.weights / sum(self.weights)#.sum()

    def _align_signatures(self, sig1: torch.Tensor, sig2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        min_len = min(len(sig1), len(sig2))
        return sig1[:min_len], sig2[:min_len]
        
    def _pad_signatures(sig1: torch.Tensor, sig2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad the smaller signature with zeros."""
        max_len = max(len(sig1), len(sig2))
        sig1_padded = torch.cat([sig1, torch.zeros(max_len - len(sig1))])
        sig2_padded = torch.cat([sig2, torch.zeros(max_len - len(sig2))])
        return sig1_padded, sig2_padded 
        
    def forward(
        self,
        X: Tuple[torch.Tensor, "PersistenceInformation"],
        Y: Tuple[torch.Tensor, "PersistenceInformation"],
    ) -> torch.Tensor:
        """Compute weighted topological signature loss between X and Y."""
        X_point_cloud, X_persistence_info = X
        Y_point_cloud, Y_persistence_info = Y

        # Compute pairwise distance matrices
        X_pairwise_dist = torch.cdist(X_point_cloud, X_point_cloud, self.p)
        Y_pairwise_dist = torch.cdist(Y_point_cloud, Y_point_cloud, self.p)

        if self.normalise:
            X_pairwise_dist = X_pairwise_dist / X_pairwise_dist.max()
            Y_pairwise_dist = Y_pairwise_dist / Y_pairwise_dist.max()

        # Compute signatures for each dimension
        XY_loss_terms = []
        YX_loss_terms = []
        
        for dim, weight in zip(self.dimensions, self.weights):
            # X's topology vs X's distances
            X_sig_X = self._select_distances(X_pairwise_dist, X_persistence_info[dim].pairing)
            # Y's topology vs X's distances
            X_sig_Y = self._select_distances(X_pairwise_dist, Y_persistence_info[dim].pairing)
            # X's topology vs Y's distances
            Y_sig_X = self._select_distances(Y_pairwise_dist, X_persistence_info[dim].pairing)
            # Y's topology vs Y's distances
            Y_sig_Y = self._select_distances(Y_pairwise_dist, Y_persistence_info[dim].pairing)

            # Align signatures before subtraction
            X_sig_X, X_sig_Y = self._align_signatures(X_sig_X, X_sig_Y)
            Y_sig_Y, Y_sig_X = self._align_signatures(Y_sig_Y, Y_sig_X)

            # Compute weighted partial distances
            XY_partial = weight * 0.5 * torch.linalg.vector_norm(X_sig_X - X_sig_Y, ord=self.p)
            YX_partial = weight * 0.5 * torch.linalg.vector_norm(Y_sig_Y - Y_sig_X, ord=self.p)

            XY_loss_terms.append(XY_partial)
            YX_loss_terms.append(YX_partial)

        # Sum all weighted terms
        loss = torch.stack(XY_loss_terms).sum() + torch.stack(YX_loss_terms).sum()
        return loss

    def _select_distances(self, dist_matrix: torch.Tensor, generators: torch.Tensor) -> torch.Tensor:
        """Select distances corresponding to topological generators."""
        if generators.shape[1] == 3:  # 0D (connected components)
            selected_dist = dist_matrix[generators[:, 1], generators[:, 2]]
        else:  # Higher dimensions (holes, voids, etc.)
            creator_dist = dist_matrix[generators[:, 0], generators[:, 1]]
            destroyer_dist = dist_matrix[generators[:, 2], generators[:, 3]]
            selected_dist = torch.abs(destroyer_dist - creator_dist)
        return selected_dist
