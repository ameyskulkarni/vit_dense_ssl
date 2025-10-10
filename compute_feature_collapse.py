import torch

# Check for feature collapse v1 (deprecated)
def compute_feature_rank_v1(features):
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


def compute_feature_rank_v2(features, eps=1e-12):
    B, H, W, D = features.shape
    flat_features = features.reshape(-1, D)

    cov = torch.cov(flat_features.T)
    eigenvals, _ = torch.linalg.eigh(cov)  # unpack directly
    eigenvals = eigenvals[eigenvals > eps]  # filter tiny values

    # Normalize eigenvalues to sum to 1
    p = eigenvals / eigenvals.sum()
    effective_rank = torch.exp(-torch.sum(p * torch.log(p + eps)))

    return effective_rank, eigenvals