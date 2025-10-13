import torch

# Check for feature collapse v1 (deprecated)
# def compute_feature_rank(features):
#     # Flatten spatial dimensions
#     B, H, W, D = features.shape
#     flat_features = features.reshape(-1, D)
#
#     # Compute covariance matrix
#     cov = torch.cov(flat_features.T)
#     eigenvals = torch.linalg.eigvals(cov).real
#
#     # Effective rank
#     eigenvals = eigenvals[eigenvals > 1e-6]
#     effective_rank = len(eigenvals)
#
#     return effective_rank, eigenvals


def compute_feature_rank(features, eps=1e-12):
    """
    Compute effective rank of features using eigenvalue entropy

    Args:
        features: Can be [B, H, W, D] (dense) or [B, D] (global)
        eps: Small epsilon for numerical stability

    Returns:
        effective_rank: Scalar tensor representing effective rank
        eigenvals: 1D tensor of eigenvalues
    """
    # Handle both dense [B, H, W, D] and global [B, D] features
    if features.dim() == 4:
        # Dense features: [B, H, W, D]
        B, H, W, D = features.shape
        flat_features = features.reshape(-1, D)
    elif features.dim() == 2:
        # Global features: [B, D]
        flat_features = features
    else:
        raise ValueError(f"Expected 2D or 4D features, got {features.dim()}D")

    # Compute covariance matrix
    cov = torch.cov(flat_features.T)

    # Get eigenvalues
    eigenvals, _ = torch.linalg.eigh(cov)

    # Filter out tiny eigenvalues (numerical noise)
    eigenvals = eigenvals[eigenvals > eps]

    # Normalize eigenvalues to sum to 1 (probability distribution)
    p = eigenvals / eigenvals.sum()

    # Compute effective rank as entropy of eigenvalue distribution
    # Higher entropy = more uniform = higher effective rank
    effective_rank = torch.exp(-torch.sum(p * torch.log(p + eps)))

    return effective_rank, eigenvals

