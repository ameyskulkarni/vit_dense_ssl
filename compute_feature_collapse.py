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


# Track parameter changes over time
def track_weight_changes(model, prev_weights):
    changes = {}
    for name, param in model.dense_projection_head.named_parameters():
        if name in prev_weights:
            change = torch.norm(param.data - prev_weights[name])
            changes[f'weight_changes/dense_head_{name}'] = change
        prev_weights[name] = param.data.clone()
    return changes, prev_weights

