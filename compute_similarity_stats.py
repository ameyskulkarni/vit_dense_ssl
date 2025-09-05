# import torch
# import torch.nn.functional as F


# def compute_similarity_stats(pos_sim, neg_sim, temperature=0.2, use_cpu=True):
#     """
#     Compute comprehensive statistics for positive and negative similarities. Comprehensive metrics tracker for dense contrastive learning
#     Helps identify if your positives/negatives are good and learning is effective
#
#     Args:
#         pos_sim: Positive similarities [B, H*W] or [B*H*W]
#         neg_sim: Negative similarities [B*H*W, queue_size]
#         temperature: Temperature used in loss computation
#         use_cpu: If True, move to CPU for memory-efficient computation
#
#     Returns:
#         Dictionary with all statistics
#     """
#
#     # Detach from autograd graph and optionally move to CPU for memory efficiency
#     with torch.no_grad():
#         if use_cpu:
#             pos_sim_detached = pos_sim.detach().cpu()
#             neg_sim_detached = neg_sim.detach().cpu()
#         else:
#             pos_sim_detached = pos_sim.detach()
#             neg_sim_detached = neg_sim.detach()
#
#     # Flatten pos_sim if needed
#     if pos_sim_detached.dim() == 2:
#         pos_sim_flat = pos_sim_detached.view(-1)  # [B*H*W]
#     else:
#         pos_sim_flat = pos_sim_detached  # Already flattened
#
#     # Compute statistics
#     stats = {}
#
#     # === POSITIVE SIMILARITIES ===
#     with torch.no_grad():
#         stats['similarity_metrics/pos_mean'] = pos_sim_flat.mean().item()
#         stats['similarity_metrics/pos_std'] = pos_sim_flat.std().item()
#         stats['similarity_metrics/pos_min'] = pos_sim_flat.min().item()
#         stats['similarity_metrics/pos_max'] = pos_sim_flat.max().item()
#         stats['similarity_metrics/pos_median'] = pos_sim_flat.median().item()
#
#         # L2 norm of the entire pos_sim tensor
#         stats['similarity_metrics/pos_l2_norm'] = pos_sim_flat.norm(p=2).item()
#         stats['similarity_metrics/pos_l1_norm'] = pos_sim_flat.norm(p=1).item()
#
#         # Number of elements
#         stats['similarity_metrics/pos_num_elements'] = pos_sim_flat.numel()
#
#     # === NEGATIVE SIMILARITIES ===
#     with torch.no_grad():
#         neg_sim_flat = neg_sim_detached.view(-1)  # Flatten all negatives
#
#         stats['similarity_metrics/neg_mean'] = neg_sim_flat.mean().item()
#         stats['similarity_metrics/neg_std'] = neg_sim_flat.std().item()
#         stats['similarity_metrics/neg_min'] = neg_sim_flat.min().item()
#         stats['similarity_metrics/neg_max'] = neg_sim_flat.max().item()
#         stats['similarity_metrics/neg_median'] = neg_sim_flat.median().item()
#
#         # L2 norm
#         stats['similarity_metrics/neg_l2_norm'] = neg_sim_flat.norm(p=2).item()
#         stats['similarity_metrics/neg_l1_norm'] = neg_sim_flat.norm(p=1).item()
#
#         # Number of elements
#         stats['similarity_metrics/neg_num_elements'] = neg_sim_flat.numel()
#
#         # # Per-query statistics (each query vs all negatives)
#         # neg_per_query_mean = neg_sim_detached.mean(dim=1)  # [B*H*W]
#         # neg_per_query_max = neg_sim_detached.max(dim=1)[0]  # [B*H*W]
#         # neg_per_query_std = neg_sim_detached.std(dim=1)  # [B*H*W]
#         #
#         # stats['similarity_metrics/neg_per_query_mean_avg'] = neg_per_query_mean.mean().item()
#         # stats['similarity_metrics/neg_per_query_max_avg'] = neg_per_query_max.mean().item()
#         # stats['similarity_metrics/neg_per_query_std_avg'] = neg_per_query_std.mean().item()
#
#     # === COMPARATIVE STATISTICS ===
#     with torch.no_grad():
#         # How much higher are positives compared to negatives?
#         stats['similarity_metrics/pos_neg_mean_diff'] = stats['similarity_metrics/pos_mean'] - stats['similarity_metrics/neg_mean']
#         stats['similarity_metrics/pos_neg_mean_ratio'] = stats['similarity_metrics/pos_mean'] / (stats['similarity_metrics/neg_mean'] + 1e-8)
#
#         # Positive vs high sim negative for each query
#         best_neg_per_query = neg_sim_detached.max(dim=1)[0]  # [B*H*W] - high sim negative for each query
#         pos_vs_best_neg = pos_sim_flat - best_neg_per_query
#
#         stats['similarity_metrics/pos_vs_best_neg_mean'] = pos_vs_best_neg.mean().item()
#         stats['similarity_metrics/pos_vs_best_neg_std'] = pos_vs_best_neg.std().item()
#         stats['similarity_metrics/pos_better_than_best_neg_pct'] = (pos_vs_best_neg > 0).float().mean().item() * 100
#
#         # # After temperature scaling (what actually goes into loss)
#         # pos_sim_temp = pos_sim_flat / temperature
#         # neg_sim_temp = neg_sim_detached / temperature
#         #
#         # stats['similarity_metrics/pos_mean_after_temp'] = pos_sim_temp.mean().item()
#         # stats['similarity_metrics/neg_mean_after_temp'] = neg_sim_temp.mean().item()
#
#     return stats


import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class ContrastiveLearningMetrics:
    """
    Comprehensive metrics tracker for dense contrastive learning
    Helps identify if your positives/negatives are good and learning is effective
    """

    def __init__(self):
        self.step = 0

    def compute_metrics(self, queries, positive_keys, neg_sim, pos_sim, correspondence,
                        corr_features_1, corr_features_2, neg_queue_features=None, temperature=0.2):
        """
        Compute comprehensive contrastive learning metrics

        Args:
            queries: [B*H*W, D] - query features
            positive_keys: [B*H*W, D] - positive key features
            neg_sim: [B*H*W, queue_size] - negative similarities
            pos_sim: [B*H*W] - positive similarities
            correspondence: [B, H*W] - correspondence indices
            corr_features_1/2: [B, H, W, D] - original spatial features
            neg_queue_features: [queue_size, D] - features in negative queue
            temperature: float - temperature parameter used in loss
        """
        metrics = {}

        # 1. SIMILARITY METRICS (Enhanced with your comprehensive stats)
        metrics.update(self._compute_similarity_metrics(pos_sim, neg_sim, temperature))

        # 2. SEPARATION METRICS
        # metrics.update(self._compute_separation_metrics(pos_sim, neg_sim))

        # 3. CORRESPONDENCE QUALITY METRICS
        metrics.update(self._compute_correspondence_metrics(
            correspondence, corr_features_1, corr_features_2))

        # 4. QUEUE DIVERSITY METRICS
        # if neg_queue_features is not None:
        #     metrics.update(self._compute_queue_metrics(queries, neg_queue_features))

        # 5. LEARNING PROGRESS METRICS
        metrics.update(self._compute_learning_metrics(queries, positive_keys))

        # Store metrics
        # for key, value in metrics.items():
        #     self.metrics_history[key].append(value)

        # self.step += 1

        return metrics

    def _compute_similarity_metrics(self, pos_sim, neg_sim, temperature=0.2):
        """Track positive and negative similarity distributions - COMPREHENSIVE VERSION"""
        with torch.no_grad():
            pos_sim_cpu = pos_sim.detach().cpu()
            neg_sim_cpu = neg_sim.detach().cpu()

            # Flatten pos_sim if needed
            if pos_sim_cpu.dim() == 2:
                pos_sim_flat = pos_sim_cpu.view(-1)  # [B*H*W]
            else:
                pos_sim_flat = pos_sim_cpu  # Already flattened

            # Flatten neg_sim for global stats
            neg_sim_flat = neg_sim_cpu.view(-1)

            stats = {}

            # === POSITIVE SIMILARITIES (Enhanced) ===
            stats['CL_similarity_metrics/pos_sim_mean'] = pos_sim_flat.mean().item()
            stats['CL_similarity_metrics/pos_sim_std'] = pos_sim_flat.std().item()
            stats['CL_similarity_metrics/pos_sim_min'] = pos_sim_flat.min().item()
            stats['CL_similarity_metrics/pos_sim_max'] = pos_sim_flat.max().item()
            stats['CL_similarity_metrics/pos_sim_median'] = pos_sim_flat.median().item()
            stats['CL_similarity_metrics/pos_l2_norm'] = pos_sim_flat.norm(p=2).item()
            stats['CL_similarity_metrics/pos_l1_norm'] = pos_sim_flat.norm(p=1).item()
            stats['CL_similarity_metrics/pos_num_elements'] = pos_sim_flat.numel()

            # === NEGATIVE SIMILARITIES (Enhanced) ===
            stats['CL_similarity_metrics/neg_sim_mean'] = neg_sim_flat.mean().item()
            stats['CL_similarity_metrics/neg_sim_std'] = neg_sim_flat.std().item()
            stats['CL_similarity_metrics/neg_sim_min'] = neg_sim_flat.min().item()
            stats['CL_similarity_metrics/neg_sim_max'] = neg_sim_flat.max().item()  # Hardest negative globally
            stats['CL_similarity_metrics/neg_sim_median'] = neg_sim_flat.median().item()
            stats['CL_similarity_metrics/neg_l2_norm'] = neg_sim_flat.norm(p=2).item()
            stats['CL_similarity_metrics/neg_l1_norm'] = neg_sim_flat.norm(p=1).item()
            stats['CL_similarity_metrics/neg_num_elements'] = neg_sim_flat.numel()

            # Top-k hard negatives
            stats['CL_similarity_metrics/neg_sim_top_k'] = neg_sim_cpu.topk(k=min(10, neg_sim_cpu.shape[1]), dim=1)[0].mean().item()

            # === COMPARATIVE STATISTICS ===
            # Global comparison
            stats['CL_similarity_metrics/pos_neg_mean_diff'] = stats['CL_similarity_metrics/pos_sim_mean'] - stats['CL_similarity_metrics/neg_sim_mean']
            stats['CL_similarity_metrics/pos_neg_mean_ratio'] = stats['CL_similarity_metrics/pos_sim_mean'] / (stats['CL_similarity_metrics/neg_sim_mean'] + 1e-8)

            # Per-query comparison (pos vs best negative for each query)
            best_neg_per_query = neg_sim_cpu.max(dim=1)[0]  # [B*H*W]
            pos_vs_best_neg = pos_sim_flat - best_neg_per_query

            stats['CL_similarity_metrics/pos_vs_best_neg_mean'] = pos_vs_best_neg.mean().item()
            stats['CL_similarity_metrics/pos_vs_best_neg_std'] = pos_vs_best_neg.std().item()
            stats['CL_similarity_metrics/pos_better_than_best_neg_pct'] = (pos_vs_best_neg > 0).float().mean().item() * 100

            # === RAW COSINE SIMILARITIES (Temperature removed) ===
            raw_pos_cosine = pos_sim_flat * temperature  # Remove temperature scaling
            raw_neg_cosine = neg_sim_flat * temperature

            stats['CL_similarity_metrics/raw_pos_cosine_mean'] = raw_pos_cosine.mean().item()
            stats['CL_similarity_metrics/raw_pos_cosine_min'] = raw_pos_cosine.min().item()
            stats['CL_similarity_metrics/raw_pos_cosine_max'] = raw_pos_cosine.max().item()
            stats['CL_similarity_metrics/raw_pos_cosine_std'] = raw_pos_cosine.std().item()

            stats['CL_similarity_metrics/raw_neg_cosine_mean'] = raw_neg_cosine.mean().item()
            stats['CL_similarity_metrics/raw_neg_cosine_max'] = raw_neg_cosine.max().item()
            stats['CL_similarity_metrics/raw_neg_cosine_min'] = raw_neg_cosine.min().item()
            stats['CL_similarity_metrics/raw_neg_cosine_std'] = raw_neg_cosine.std().item()

            # Correspondence quality indicators
            stats['CL_similarity_metrics/good_correspondences_rate'] = (raw_pos_cosine > 0.5).float().mean().item()
            stats['CL_similarity_metrics/bad_correspondences_rate'] = (raw_pos_cosine < 0.0).float().mean().item()
            stats['CL_similarity_metrics/excellent_correspondences_rate'] = (raw_pos_cosine > 0.8).float().mean().item()

            stats['CL_similarity_metrics/good_neg_correspondences_rate'] = ((raw_neg_cosine > -0.2) & (raw_neg_cosine < 0.2)).float().mean().item()
            stats['CL_similarity_metrics/bad_neg_correspondences_rate'] = ((raw_neg_cosine < -0.2) | (raw_neg_cosine > 0.2)).float().mean().item()
            stats['CL_similarity_metrics/excellent_neg_correspondences_rate'] = ((raw_neg_cosine > -0.1) & (raw_neg_cosine < 0.1)).float().mean().item()

            return stats

    def _compute_separation_metrics(self, pos_sim, neg_sim):
        """Measure how well positives are separated from negatives"""
        with torch.no_grad():
            pos_sim_cpu = pos_sim.detach().cpu()
            neg_sim_cpu = neg_sim.detach().cpu()

            print(f"Shape of pos_sim and neg_sim: {pos_sim_cpu.shape}, {neg_sim_cpu.shape}")

            # Hard negative (max negative similarity for each query)
            hard_neg_sim = neg_sim_cpu.max(dim=1)[0]

            # Separation gap (positive should be higher than hardest negative)
            separation_gap = pos_sim_cpu - hard_neg_sim

            # Ranking metrics
            neg_higher_than_pos = (hard_neg_sim > pos_sim_cpu).float().mean()

            # Top-k negative comparison
            top_k_neg = neg_sim_cpu.topk(k=5, dim=1)[0].mean(dim=1)
            top_k_separation = pos_sim_cpu - top_k_neg

            return {
                'CL_similarity_metrics/separation_gap_mean': separation_gap.mean().item(),
                'CL_similarity_metrics/separation_gap_std': separation_gap.std().item(),
                'CL_similarity_metrics/negative_higher_than_positive_rate': neg_higher_than_pos.item(),
                'CL_similarity_metrics/top_k_separation_mean': top_k_separation.mean().item(),
                'CL_similarity_metrics/positive_rank_in_negatives': self._compute_positive_rank(pos_sim_cpu, neg_sim_cpu),
            }

    def _compute_positive_rank(self, pos_sim, neg_sim):
        """Compute average rank of positive among negatives (lower is better)"""
        # For each query, count how many negatives have higher similarity than positive
        ranks = (neg_sim > pos_sim.unsqueeze(1)).sum(dim=1).float() + 1
        return ranks.mean().item()

    def _compute_correspondence_metrics(self, correspondence, features_1, features_2):
        """Evaluate quality of dense correspondences"""
        with torch.no_grad():
            B, H, W, D = features_1.shape

            # Correspondence consistency (do correspondences make sense?)
            f1_flat = features_1.view(B, H * W, D)
            f2_flat = features_2.view(B, H * W, D)

            # Get actual corresponding features
            batch_idx = torch.arange(B).unsqueeze(1).expand(-1, H * W).to(correspondence.device)
            corresponding_features = f2_flat[batch_idx, correspondence]

            # Cosine similarity between matched features
            f1_norm = F.normalize(f1_flat, dim=2)
            corresp_norm = F.normalize(corresponding_features, dim=2)
            correspondence_quality = (f1_norm * corresp_norm).sum(dim=2)

            # Correspondence entropy (diversity of correspondences)
            correspondence_hist = torch.histc(correspondence.float(), bins=min(100, H * W), min=0, max=H * W - 1)
            correspondence_entropy = -(correspondence_hist / correspondence_hist.sum() *
                                       torch.log(correspondence_hist / correspondence_hist.sum() + 1e-8)).sum()

            return {
                'CL_similarity_metrics/correspondence_quality_mean': correspondence_quality.mean().item(),
                'CL_similarity_metrics/correspondence_quality_std': correspondence_quality.std().item(),
                'CL_similarity_metrics/correspondence_entropy': correspondence_entropy.item(),
                'CL_similarity_metrics/correspondence_diversity': (correspondence_hist > 0).float().mean().item(),
            }

    def _compute_queue_metrics(self, queries, neg_queue_features):
        """Analyze negative queue quality and diversity"""
        with torch.no_grad():
            # Queue diversity (how diverse are the negative features?)
            queue_norm = F.normalize(neg_queue_features, dim=1)

            # Pairwise similarities in queue
            queue_sim_matrix = torch.mm(queue_norm, queue_norm.t())

            # Remove diagonal (self-similarity)
            mask = ~torch.eye(queue_sim_matrix.size(0), dtype=torch.bool, device=queue_sim_matrix.device)
            queue_similarities = queue_sim_matrix[mask]

            # Query-queue similarity distribution
            query_norm = F.normalize(queries, dim=1)
            query_queue_sim = torch.mm(query_norm, queue_norm.t())

            return {
                'CL_similarity_metrics/queue_internal_similarity_mean': queue_similarities.mean().item(),
                'CL_similarity_metrics/queue_internal_similarity_std': queue_similarities.std().item(),
                'CL_similarity_metrics/query_queue_similarity_mean': query_queue_sim.mean().item(),
                'CL_similarity_metrics/query_queue_similarity_max': query_queue_sim.max().item(),
                'CL_similarity_metrics/queue_diversity_score': 1.0 - queue_similarities.mean().item(),  # Higher is more diverse
            }

    def _compute_learning_metrics(self, queries, positive_keys):
        """Track representation learning progress"""
        with torch.no_grad():
            # Feature norm stability
            query_norms = queries.norm(dim=1)
            key_norms = positive_keys.norm(dim=1)

            # Feature dimensionality usage (how much of the embedding space is used?)
            query_std_per_dim = queries.std(dim=0).mean()

            return {
                'CL_similarity_metrics/query_norm_mean': query_norms.mean().item(),
                'CL_similarity_metrics/query_norm_std': query_norms.std().item(),
                'CL_similarity_metrics/key_norm_mean': key_norms.mean().item(),
                'CL_similarity_metrics/feature_dimensionality_usage': query_std_per_dim.item(),
            }

    # def _log_metrics(self, current_metrics):
    #     """Log current metrics"""
    #     print(f"\n=== Contrastive Learning Metrics (Step {self.step}) ===")
    #
    #     # Core metrics to watch
    #     print(f"🎯 Pos Sim Mean (scaled): {current_metrics['pos_sim_mean']:.4f}")
    #     print(f"🎯 Pos Sim Mean (raw cosine): {current_metrics['raw_pos_cosine_mean']:.4f}")
    #     print(f"❌ Neg Sim Mean (scaled): {current_metrics['neg_sim_mean']:.4f}")
    #     print(f"❌ Neg Sim Mean (raw cosine): {current_metrics['raw_neg_cosine_mean']:.4f}")
    #     print(f"📏 Separation Gap: {current_metrics['separation_gap_mean']:.4f}")
    #     print(f"🏆 Positive Rank: {current_metrics['positive_rank_in_negatives']:.1f}")
    #     print(f"⚠️  Neg > Pos Rate: {current_metrics['negative_higher_than_positive_rate']:.4f}")
    #     print(f"✅ Pos Better Than Best Neg: {current_metrics['pos_better_than_best_neg_pct']:.1f}%")
    #
    #     # Correspondence quality
    #     print(f"🎯 Good Correspondences (>0.5): {current_metrics['good_correspondences_rate']:.4f}")
    #     print(f"⭐ Excellent Correspondences (>0.8): {current_metrics['excellent_correspondences_rate']:.4f}")
    #     print(f"❌ Bad Correspondences (<0.0): {current_metrics['bad_correspondences_rate']:.4f}")
    #
    #     if 'queue_diversity_score' in current_metrics:
    #         print(f"🔄 Queue Diversity: {current_metrics['queue_diversity_score']:.4f}")
    #
    #     print(f"📊 Correspondence Quality: {current_metrics['correspondence_quality_mean']:.4f}")
    #
    #     # Raw similarity ranges
    #     print(
    #         f"📈 Raw Pos Cosine Range: [{current_metrics['raw_pos_cosine_min']:.3f}, {current_metrics['raw_pos_cosine_max']:.3f}]")
    #     print(f"📈 Scaled Pos Sim Range: [{current_metrics['pos_sim_min']:.1f}, {current_metrics['pos_sim_max']:.1f}]")

    # def get_health_report(self):
    #     """Generate a health report of contrastive learning"""
    #     if not self.metrics_history:
    #         return "No metrics collected yet"
    #
    #     recent_window = min(self.moving_avg_window, len(self.metrics_history['pos_sim_mean']))
    #
    #     report = []
    #     report.append("🏥 CONTRASTIVE LEARNING HEALTH REPORT")
    #     report.append("=" * 50)
    #
    #     # 1. Similarity Health (Enhanced with raw cosine similarities)
    #     pos_sim = np.array(self.metrics_history['pos_sim_mean'][-recent_window:]).mean()
    #     neg_sim = np.array(self.metrics_history['neg_sim_mean'][-recent_window:]).mean()
    #     separation = np.array(self.metrics_history['separation_gap_mean'][-recent_window:]).mean()
    #
    #     # Raw cosine similarities
    #     raw_pos_cosine = np.array(self.metrics_history['raw_pos_cosine_mean'][-recent_window:]).mean()
    #     raw_neg_cosine = np.array(self.metrics_history['raw_neg_cosine_mean'][-recent_window:]).mean()
    #
    #     report.append(f"📊 SIMILARITY ANALYSIS:")
    #     report.append(f"   Positive Similarity (scaled): {pos_sim:.4f}")
    #     report.append(f"   Positive Similarity (raw cosine): {raw_pos_cosine:.4f}")
    #     report.append(f"   Negative Similarity (scaled): {neg_sim:.4f}")
    #     report.append(f"   Negative Similarity (raw cosine): {raw_neg_cosine:.4f}")
    #     report.append(f"   Separation Gap: {separation:.4f}")
    #
    #     if separation > 1.0:
    #         report.append("   ✅ Good separation - positives clearly higher than negatives")
    #     elif separation > 0.5:
    #         report.append("   ⚠️  Moderate separation - could be improved")
    #     else:
    #         report.append("   ❌ Poor separation - negatives competing with positives")
    #
    #     # 2. Ranking Health
    #     pos_rank = np.array(self.metrics_history['positive_rank_in_negatives'][-recent_window:]).mean()
    #     neg_higher_rate = np.array(self.metrics_history['negative_higher_than_positive_rate'][-recent_window:]).mean()
    #     pos_better_pct = np.array(self.metrics_history['pos_better_than_best_neg_pct'][-recent_window:]).mean()
    #
    #     report.append(f"\n🏆 RANKING ANALYSIS:")
    #     report.append(f"   Average Positive Rank: {pos_rank:.1f}")
    #     report.append(f"   Negatives > Positives Rate: {neg_higher_rate:.4f}")
    #     report.append(f"   Positives Better Than Best Negative: {pos_better_pct:.1f}%")
    #
    #     if pos_rank < 5 and neg_higher_rate < 0.1:
    #         report.append("   ✅ Excellent ranking - positives consistently top-ranked")
    #     elif pos_rank < 20 and neg_higher_rate < 0.3:
    #         report.append("   ⚠️  Good ranking with room for improvement")
    #     else:
    #         report.append("   ❌ Poor ranking - many negatives rank higher than positives")
    #
    #     # 3. Queue Health (if available)
    #     if 'queue_diversity_score' in self.metrics_history:
    #         queue_diversity = np.array(self.metrics_history['queue_diversity_score'][-recent_window:]).mean()
    #
    #         report.append(f"\n🔄 QUEUE ANALYSIS:")
    #         report.append(f"   Queue Diversity Score: {queue_diversity:.4f}")
    #
    #         if queue_diversity > 0.7:
    #             report.append("   ✅ Diverse queue - good negative sampling")
    #         elif queue_diversity > 0.5:
    #             report.append("   ⚠️  Moderate diversity - consider queue size increase")
    #         else:
    #             report.append("   ❌ Low diversity - queue too small or repetitive")
    #
    #     # 4. Correspondence Health (Enhanced)
    #     corresp_quality = np.array(self.metrics_history['correspondence_quality_mean'][-recent_window:]).mean()
    #     good_corresp_rate = np.array(self.metrics_history['good_correspondences_rate'][-recent_window:]).mean()
    #     excellent_corresp_rate = np.array(
    #         self.metrics_history['excellent_correspondences_rate'][-recent_window:]).mean()
    #     bad_corresp_rate = np.array(self.metrics_history['bad_correspondences_rate'][-recent_window:]).mean()
    #
    #     report.append(f"\n🎯 CORRESPONDENCE ANALYSIS:")
    #     report.append(f"   Correspondence Quality: {corresp_quality:.4f}")
    #     report.append(f"   Good Correspondences (>0.5 cosine): {good_corresp_rate:.4f}")
    #     report.append(f"   Excellent Correspondences (>0.8 cosine): {excellent_corresp_rate:.4f}")
    #     report.append(f"   Bad Correspondences (<0.0 cosine): {bad_corresp_rate:.4f}")
    #
    #     if good_corresp_rate > 0.7 and bad_corresp_rate < 0.1:
    #         report.append("   ✅ High-quality correspondences")
    #     elif good_corresp_rate > 0.5 and bad_corresp_rate < 0.2:
    #         report.append("   ⚠️  Moderate correspondence quality")
    #     else:
    #         report.append("   ❌ Poor correspondences - check augmentation strategy")
    #
    #     # 5. Overall Assessment
    #     report.append(f"\n🎯 OVERALL ASSESSMENT:")
    #
    #     scores = []
    #     if separation > 0.5: scores.append(1)
    #     if pos_rank < 20: scores.append(1)
    #     if neg_higher_rate < 0.3: scores.append(1)
    #     if good_corresp_rate > 0.5: scores.append(1)
    #     if bad_corresp_rate < 0.2: scores.append(1)
    #
    #     health_score = sum(scores) / 5
    #
    #     if health_score >= 0.8:
    #         report.append("   ✅ HEALTHY - Contrastive learning is working well!")
    #     elif health_score >= 0.6:
    #         report.append("   ⚠️  MODERATE - Some improvements needed")
    #     else:
    #         report.append("   ❌ UNHEALTHY - Significant issues detected")
    #
    #     return "\n".join(report)
