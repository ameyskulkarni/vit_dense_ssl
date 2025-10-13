import torch
import numpy as np

class ContrastiveLearningMetrics:
    """
    Comprehensive metrics tracker for dense contrastive learning
    Helps identify if your positives/negatives are good and learning is effective
    """

    def __init__(self):
        self.step = 0

    def compute_metrics(self, neg_sim, pos_sim, temperature=0.2):
        """
        Compute comprehensive contrastive learning metrics

        Args:
            neg_sim: [B*H*W, queue_size] - negative similarities
            pos_sim: [B*H*W] - positive similarities
            temperature: float - temperature parameter used in loss
        """
        metrics = {}

        # 1. SIMILARITY METRICS
        metrics.update(self._compute_similarity_metrics(pos_sim, neg_sim, temperature))


        return metrics

    def _compute_similarity_metrics(self, pos_sim, neg_sim, temperature=0.2):
        """Track positive and negative similarity distributions - CORRECTED PER-QUERY VERSION"""
        with torch.no_grad():
            pos_sim_cpu = pos_sim.detach().cpu()
            neg_sim_cpu = neg_sim.detach().cpu()  # [B*H*W, queue_size]

            # Flatten pos_sim if needed
            if pos_sim_cpu.dim() == 2:
                pos_sim_flat = pos_sim_cpu.view(-1)  # [B*H*W]
            else:
                pos_sim_flat = pos_sim_cpu  # Already flattened

            B_HW, queue_size = neg_sim_cpu.shape
            stats = {}

            # === POSITIVE SIMILARITIES (Same as before - already correct) ===
            stats['CL_similarity_metrics/pos_sim_mean'] = pos_sim_flat.mean().item()
            stats['CL_similarity_metrics/pos_sim_std'] = pos_sim_flat.std().item()
            stats['CL_similarity_metrics/pos_sim_min'] = pos_sim_flat.min().item()
            stats['CL_similarity_metrics/pos_sim_max'] = pos_sim_flat.max().item()
            stats['CL_similarity_metrics/pos_sim_median'] = pos_sim_flat.median().item()

            # === NEGATIVE SIMILARITIES (CORRECTED - Per-query analysis) ===

            # Calculate per-query negative statistics
            neg_means_per_query = neg_sim_cpu.mean(dim=1)  # [B*H*W] - mean negative sim per query
            neg_stds_per_query = neg_sim_cpu.std(dim=1)  # [B*H*W] - std of negatives per query
            neg_mins_per_query = neg_sim_cpu.min(dim=1)[0]  # [B*H*W] - min negative per query
            neg_maxs_per_query = neg_sim_cpu.max(dim=1)[0]  # [B*H*W] - max (hardest) negative per query
            neg_medians_per_query = neg_sim_cpu.median(dim=1)[0]  # [B*H*W] - median negative per query

            # Aggregate per-query statistics across all queries
            stats['CL_similarity_metrics/neg_sim_mean'] = neg_means_per_query.mean().item()  # Average of per-query means
            stats['CL_similarity_metrics/neg_mean_std_across_queries'] = neg_means_per_query.std().item()  # How varied are query means
            stats['CL_similarity_metrics/neg_sim_std'] = neg_stds_per_query.mean().item()  # Average of per-query stds
            stats['CL_similarity_metrics/neg_sim_min_mean'] = neg_mins_per_query.mean().item()  # minimum mean
            stats['CL_similarity_metrics/neg_sim_max_mean'] = neg_maxs_per_query.mean().item()  # maximum mean
            stats['CL_similarity_metrics/neg_min_std_across_queries'] = neg_mins_per_query.std().item()  # Consistency of hardest negatives
            stats['CL_similarity_metrics/neg_max_std_across_queries'] = neg_maxs_per_query.std().item()  # Consistency of hardest negatives
            stats['CL_similarity_metrics/neg_sim_median_mean'] = neg_medians_per_query.mean().item()  # Average of per-query medians

            # Top-k hard negatives per query (CORRECTED)
            k = min(10, queue_size)
            top_10_negs_per_query = neg_sim_cpu.topk(k=k, dim=1)[0]  # [B*H*W, k]
            stats['CL_similarity_metrics/neg_sim_top_k'] = top_10_negs_per_query.mean().item()  # Average of top-k across queries


            # === COMPARATIVE STATISTICS (CORRECTED) ===
            # Global comparison - now using corrected negative means
            stats['CL_similarity_metrics/pos_neg_mean_diff'] = stats['CL_similarity_metrics/pos_sim_mean'] - stats[
                'CL_similarity_metrics/neg_sim_mean']
            stats['CL_similarity_metrics/pos_neg_mean_ratio'] = stats['CL_similarity_metrics/pos_sim_mean'] / (
                        stats['CL_similarity_metrics/neg_sim_mean'] + 1e-8)

            # Per-query comparison (pos vs best negative for each query) - This was already correct!
            best_neg_per_query = neg_maxs_per_query  # [B*H*W] - reuse calculated values
            pos_vs_best_neg = pos_sim_flat - best_neg_per_query

            stats['CL_similarity_metrics/pos_vs_best_neg_mean'] = pos_vs_best_neg.mean().item()
            stats['CL_similarity_metrics/pos_vs_best_neg_std'] = pos_vs_best_neg.std().item()
            stats['CL_similarity_metrics/pos_better_than_best_neg_pct'] = (pos_vs_best_neg > 0).float().mean().item() * 100

            # === RAW COSINE SIMILARITIES (Temperature removed) ===
            raw_pos_cosine = pos_sim_flat * temperature  # Remove temperature scaling

            # Per-query raw negative cosines
            raw_neg_cosine_per_query = neg_sim_cpu * temperature  # [B*H*W, queue_size]
            raw_neg_means_per_query = raw_neg_cosine_per_query.mean(dim=1)  # [B*H*W]


            # === CORRESPONDENCE QUALITY INDICATORS (Enhanced) ===

            # Positive correspondence quality
            stats['CL_similarity_metrics/good_correspondences_rate'] = (raw_pos_cosine > 0.5).float().mean().item()
            stats['CL_similarity_metrics/bad_correspondences_rate'] = (raw_pos_cosine < 0.0).float().mean().item()
            stats['CL_similarity_metrics/excellent_correspondences_rate'] = (raw_pos_cosine > 0.8).float().mean().item()
            stats['CL_similarity_metrics/very_bad_correspondences_rate'] = (raw_pos_cosine < -0.2).float().mean().item()

            # Negative correspondence quality (per-query analysis)
            # For negatives, we want them to be dissimilar (negative cosine or small positive)
            # Negative pool quality analysis (CORRECTED: lower cosine = better negatives)

            # Average negatives quality (per query average negative similarity)
            excellent_avg_negs = raw_neg_means_per_query < -0.2  # Very dissimilar on average
            good_avg_negs = (raw_neg_means_per_query >= -0.2) & (raw_neg_means_per_query < 0.0)
            okay_avg_negs = (raw_neg_means_per_query >= 0.0) & (raw_neg_means_per_query <= 0.2)
            bad_avg_negs = raw_neg_means_per_query > 0.2  # Too similar on average

            stats['CL_similarity_metrics/excellent_avg_negatives_rate'] = excellent_avg_negs.float().mean().item()
            stats['CL_similarity_metrics/good_avg_negatives_rate'] = good_avg_negs.float().mean().item()
            stats['CL_similarity_metrics/okay_avg_negatives_rate'] = okay_avg_negs.float().mean().item()
            stats['CL_similarity_metrics/bad_avg_negatives_rate'] = bad_avg_negs.float().mean().item()

            return stats