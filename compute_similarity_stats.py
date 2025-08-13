import torch
import torch.nn.functional as F


def compute_similarity_stats(pos_sim, neg_sim, temperature=0.2, use_cpu=True):
    """
    Compute comprehensive statistics for positive and negative similarities

    Args:
        pos_sim: Positive similarities [B, H*W] or [B*H*W]
        neg_sim: Negative similarities [B*H*W, queue_size]
        temperature: Temperature used in loss computation
        use_cpu: If True, move to CPU for memory-efficient computation

    Returns:
        Dictionary with all statistics
    """

    # Detach from autograd graph and optionally move to CPU for memory efficiency
    with torch.no_grad():
        if use_cpu:
            pos_sim_detached = pos_sim.detach().cpu()
            neg_sim_detached = neg_sim.detach().cpu()
        else:
            pos_sim_detached = pos_sim.detach()
            neg_sim_detached = neg_sim.detach()

    # Flatten pos_sim if needed
    if pos_sim_detached.dim() == 2:
        pos_sim_flat = pos_sim_detached.view(-1)  # [B*H*W]
    else:
        pos_sim_flat = pos_sim_detached  # Already flattened

    # Compute statistics
    stats = {}

    # === POSITIVE SIMILARITIES ===
    with torch.no_grad():
        stats['similarity_metrics/pos_mean'] = pos_sim_flat.mean().item()
        stats['similarity_metrics/pos_std'] = pos_sim_flat.std().item()
        stats['similarity_metrics/pos_min'] = pos_sim_flat.min().item()
        stats['similarity_metrics/pos_max'] = pos_sim_flat.max().item()
        stats['similarity_metrics/pos_median'] = pos_sim_flat.median().item()

        # L2 norm of the entire pos_sim tensor
        stats['similarity_metrics/pos_l2_norm'] = pos_sim_flat.norm(p=2).item()
        stats['similarity_metrics/pos_l1_norm'] = pos_sim_flat.norm(p=1).item()

        # Number of elements
        stats['similarity_metrics/pos_num_elements'] = pos_sim_flat.numel()

    # === NEGATIVE SIMILARITIES ===
    with torch.no_grad():
        neg_sim_flat = neg_sim_detached.view(-1)  # Flatten all negatives

        stats['similarity_metrics/neg_mean'] = neg_sim_flat.mean().item()
        stats['similarity_metrics/neg_std'] = neg_sim_flat.std().item()
        stats['similarity_metrics/neg_min'] = neg_sim_flat.min().item()
        stats['similarity_metrics/neg_max'] = neg_sim_flat.max().item()
        stats['similarity_metrics/neg_median'] = neg_sim_flat.median().item()

        # L2 norm
        stats['similarity_metrics/neg_l2_norm'] = neg_sim_flat.norm(p=2).item()
        stats['similarity_metrics/neg_l1_norm'] = neg_sim_flat.norm(p=1).item()

        # Number of elements
        stats['similarity_metrics/neg_num_elements'] = neg_sim_flat.numel()

        # # Per-query statistics (each query vs all negatives)
        # neg_per_query_mean = neg_sim_detached.mean(dim=1)  # [B*H*W]
        # neg_per_query_max = neg_sim_detached.max(dim=1)[0]  # [B*H*W]
        # neg_per_query_std = neg_sim_detached.std(dim=1)  # [B*H*W]
        #
        # stats['similarity_metrics/neg_per_query_mean_avg'] = neg_per_query_mean.mean().item()
        # stats['similarity_metrics/neg_per_query_max_avg'] = neg_per_query_max.mean().item()
        # stats['similarity_metrics/neg_per_query_std_avg'] = neg_per_query_std.mean().item()

    # === COMPARATIVE STATISTICS ===
    with torch.no_grad():
        # How much higher are positives compared to negatives?
        stats['similarity_metrics/pos_neg_mean_diff'] = stats['similarity_metrics/pos_mean'] - stats['similarity_metrics/neg_mean']
        stats['similarity_metrics/pos_neg_mean_ratio'] = stats['similarity_metrics/pos_mean'] / (stats['similarity_metrics/neg_mean'] + 1e-8)

        # Positive vs high sim negative for each query
        best_neg_per_query = neg_sim_detached.max(dim=1)[0]  # [B*H*W] - high sim negative for each query
        pos_vs_best_neg = pos_sim_flat - best_neg_per_query

        stats['similarity_metrics/pos_vs_best_neg_mean'] = pos_vs_best_neg.mean().item()
        stats['similarity_metrics/pos_vs_best_neg_std'] = pos_vs_best_neg.std().item()
        stats['similarity_metrics/pos_better_than_best_neg_pct'] = (pos_vs_best_neg > 0).float().mean().item() * 100

        # # After temperature scaling (what actually goes into loss)
        # pos_sim_temp = pos_sim_flat / temperature
        # neg_sim_temp = neg_sim_detached / temperature
        # 
        # stats['similarity_metrics/pos_mean_after_temp'] = pos_sim_temp.mean().item()
        # stats['similarity_metrics/neg_mean_after_temp'] = neg_sim_temp.mean().item()

    return stats
