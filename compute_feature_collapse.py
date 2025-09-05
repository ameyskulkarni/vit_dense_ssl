import torch

# Check for feature collapse
def compute_feature_rank(features):
    # Flatten spatial dimensions
    B, H, W, D = features.shape
    flat_features = features.reshape(-1, D)

    # Compute covariance matrix
    cov = torch.cov(flat_features.T)
    eigenvals = torch.linalg.eigvals(cov).real

    # Effective rank
    eigenvals = eigenvals[eigenvals > 1e-6]
    effective_rank = len(eigenvals)

    return effective_rank, eigenvals
