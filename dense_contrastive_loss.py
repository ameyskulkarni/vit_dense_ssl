import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random

class DenseContrastiveLoss(nn.Module):
    """
    Dense Contrastive Loss as described in DenseCL paper
    """

    def __init__(self, temperature=0.2, queue_size=65536, momentum=0.999, correspondence_features='dense', max_patches_per_image=50,
                 sampling_strategy='random'):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum
        self.correspondence_features = correspondence_features
        self.max_patches_per_image = max_patches_per_image
        self.sampling_strategy = sampling_strategy  # 'random', 'diverse', or 'hardest'

        # Initialize memory queue
        self.register_buffer("queue", torch.randn(queue_size, 128))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Track image diversity in queue
        self.register_buffer("image_ids", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("current_image_id", torch.zeros(1, dtype=torch.long))

        # Normalize the queue
        self.queue = F.normalize(self.queue, dim=1)

    def _sample_patches_strategically(self, features, batch_size, height, width):
        """
        Sample patches strategically to maximize diversity
        Args:
            features: [B, H*W, D] flattened features
            batch_size, height, width: dimensions
        Returns:
            sampled_features: [B, max_patches_per_image, D]
            sampled_indices: [B, max_patches_per_image] indices of sampled patches
        """
        B, HW, D = features.shape
        device = features.device

        sampled_features_list = []
        sampled_indices_list = []

        for b in range(B):
            patch_features = features[b]  # [H*W, D]

            if self.sampling_strategy == 'random':
                # Simple random sampling
                if HW <= self.max_patches_per_image:
                    indices = torch.arange(HW, device=device)
                    sampled_patches = patch_features
                else:
                    indices = torch.randperm(HW, device=device)[:self.max_patches_per_image]
                    sampled_patches = patch_features[indices]

            elif self.sampling_strategy == 'diverse':
                # Diverse sampling using clustering-like approach
                sampled_patches, indices = self._diverse_sampling(patch_features, HW, device)

            elif self.sampling_strategy == 'hardest':
                # Sample patches that are most different from current queue
                sampled_patches, indices = self._hard_negative_sampling(patch_features, HW, device)

            # Pad if necessary
            if len(indices) < self.max_patches_per_image:
                padding_needed = self.max_patches_per_image - len(indices)
                # Repeat random patches to fill
                repeat_indices = torch.randint(0, len(indices), (padding_needed,), device=device)
                indices = torch.cat([indices, indices[repeat_indices]])
                sampled_patches = torch.cat([sampled_patches, sampled_patches[repeat_indices]])

            sampled_features_list.append(sampled_patches)
            sampled_indices_list.append(indices)

        sampled_features = torch.stack(sampled_features_list)  # [B, max_patches_per_image, D]
        sampled_indices = torch.stack(sampled_indices_list)  # [B, max_patches_per_image]

        return sampled_features, sampled_indices

    def _diverse_sampling(self, patch_features, HW, device):
        """Diverse sampling using k-means-like approach"""
        if HW <= self.max_patches_per_image:
            indices = torch.arange(HW, device=device)
            return patch_features, indices

        # Simple diverse sampling: divide space into grid and sample from each region
        H = W = int(HW ** 0.5)  # Assume square feature map
        grid_size = int((self.max_patches_per_image) ** 0.5)

        indices = []
        step_h = max(1, H // grid_size)
        step_w = max(1, W // grid_size)

        for i in range(0, H, step_h):
            for j in range(0, W, step_w):
                if len(indices) >= self.max_patches_per_image:
                    break
                # Add some randomness within each grid cell
                actual_i = min(H - 1, i + random.randint(0, min(step_h - 1, H - 1 - i)))
                actual_j = min(W - 1, j + random.randint(0, min(step_w - 1, W - 1 - j)))
                idx = actual_i * W + actual_j
                indices.append(idx)

        # Fill remaining with random samples
        while len(indices) < self.max_patches_per_image:
            idx = random.randint(0, HW - 1)
            if idx not in indices:
                indices.append(idx)

        indices = torch.tensor(indices[:self.max_patches_per_image], device=device)
        sampled_patches = patch_features[indices]

        return sampled_patches, indices

    def _hard_negative_sampling(self, patch_features, HW, device):
        """Sample patches that are most dissimilar to current queue"""
        if HW <= self.max_patches_per_image:
            indices = torch.arange(HW, device=device)
            return patch_features, indices

        # Compute similarity with queue and select most dissimilar
        with torch.no_grad():
            queue_normalized = F.normalize(self.queue, dim=1)
            patch_normalized = F.normalize(patch_features, dim=1)

            # Similarity to queue: [H*W, queue_size]
            sim_to_queue = torch.mm(patch_normalized, queue_normalized.t())

            # Average similarity to queue for each patch
            avg_sim = sim_to_queue.mean(dim=1)  # [H*W]

            # Select patches with lowest average similarity (hardest negatives)
            _, indices = torch.topk(-avg_sim, self.max_patches_per_image, sorted=False)

        sampled_patches = patch_features[indices]
        return sampled_patches, indices

    def _dequeue_and_enqueue_diverse(self, keys, batch_size):
        """
        Update the memory queue with diverse negative samples
        Args:
            keys: [B, max_patches_per_image, D] sampled keys
            batch_size: number of images in batch
        """
        # Flatten the sampled keys
        keys_flat = keys.view(-1, keys.size(-1))  # [B * max_patches_per_image, D]
        total_samples = keys_flat.size(0)

        ptr = int(self.queue_ptr)

        # Update image IDs for tracking diversity
        current_id = int(self.current_image_id)

        # Handle case where we have more samples than queue size
        if total_samples >= self.queue_size:
            # Randomly sample from the new keys to fill entire queue
            random_indices = torch.randperm(total_samples)[:self.queue_size]
            self.queue.copy_(keys_flat[random_indices])

            # Update image IDs (each image gets multiple slots)
            samples_per_image = self.queue_size // batch_size
            for i in range(batch_size):
                start_idx = i * samples_per_image
                end_idx = (i + 1) * samples_per_image if i < batch_size - 1 else self.queue_size
                self.image_ids[start_idx:end_idx] = current_id + i

            self.queue_ptr[0] = 0
        else:
            # Handle wrap-around case
            if ptr + total_samples > self.queue_size:
                remaining_space = self.queue_size - ptr

                # Fill remaining space at the end
                self.queue[ptr:].copy_(keys_flat[:remaining_space])

                # Put the rest at the beginning
                overflow = total_samples - remaining_space
                self.queue[:overflow].copy_(keys_flat[remaining_space:])

                # Update image IDs
                self.image_ids[ptr:] = current_id
                self.image_ids[:overflow] = current_id

                ptr = overflow
            else:
                # Normal case
                self.queue[ptr:ptr + total_samples].copy_(keys_flat)
                self.image_ids[ptr:ptr + total_samples] = current_id
                ptr = ptr + total_samples

            self.queue_ptr[0] = ptr

        # Increment image ID for next batch
        self.current_image_id[0] = current_id + batch_size

    def get_queue_diversity_stats(self):
        """Get statistics about queue diversity"""
        unique_images = torch.unique(self.image_ids).size(0)
        total_slots = self.queue_size
        return {
            'unique_images_in_queue': unique_images,
            'total_queue_slots': total_slots,
            'diversity_ratio': unique_images / total_slots
        }

    def _dequeue_and_enqueue(self, keys):
        """Update the memory queue with new keys - FIXED VERSION"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Handle case where batch_size > queue_size
        if batch_size >= self.queue_size:
            # If we have more keys than queue size, just take the last queue_size keys
            self.queue.copy_(keys[-self.queue_size:])
            self.queue_ptr[0] = 0
            return

        # Handle wrap-around case
        if ptr + batch_size > self.queue_size:
            # Split the batch: some goes to end of queue, rest goes to beginning
            remaining_space = self.queue_size - ptr

            # Fill the remaining space at the end
            self.queue[ptr:].copy_(keys[:remaining_space])

            # Put the rest at the beginning
            overflow = batch_size - remaining_space
            self.queue[:overflow].copy_(keys[remaining_space:])

            # Update pointer
            ptr = overflow
        else:
            # Normal case: enough space in queue
            self.queue[ptr:ptr + batch_size].copy_(keys)
            ptr = ptr + batch_size

        self.queue_ptr[0] = ptr

    def extract_correspondence(self, f1, f2):
        """
        Extract dense correspondence between two views
        Args:
            f1, f2: Feature maps [B, H, W, D]
        Returns:
            correspondence: Matching indices [B, H*W]
        """
        B, H, W, D = f1.shape

        # Flatten spatial dimensions
        f1_flat = f1.view(B, H * W, D)  # [B, H*W, D]
        f2_flat = f2.view(B, H * W, D)  # [B, H*W, D]

        # Compute cosine similarity matrix
        f1_norm = F.normalize(f1_flat, dim=2)
        f2_norm = F.normalize(f2_flat, dim=2)

        # Similarity matrix: [B, H*W, H*W]
        # We are calculating similarity using dot product, not using cosine similarity. It may be prudent to
        # calculate similarity using cosine similarity or other similarity measures.
        sim_matrix = torch.bmm(f1_norm, f2_norm.transpose(1, 2))

        # Find correspondence (argmax along last dimension)
        correspondence = torch.argmax(sim_matrix, dim=2)  # [B, H*W]

        return correspondence

    def forward(self, dense_features_1, dense_features_2, backbone_features_1, backbone_features_2):
        """
        Compute dense contrastive loss
        Args:
            dense_features_1, dense_features_2: Dense features from projection head [B, H, W, D]
            backbone_features_1, backbone_features_2: Backbone features for correspondence [B, H, W, D]
        """
        B, H, W, D = dense_features_1.shape

        # Extract correspondence using self.correspondense_features
        if self.correspondence_features == 'dense':
            correspondence = self.extract_correspondence(dense_features_1, dense_features_2)
        elif self.correspondence_features == 'backbone':
            correspondence = self.extract_correspondence(backbone_features_1, backbone_features_2)
        else:
            print(f"Unknown correspondence features. Specify 'dense' or 'backbone'")
            sys.exit(1)

        # Flatten dense features
        queries = dense_features_1.view(B, H * W, D)  # [B, H*W, D]
        keys = dense_features_2.view(B, H * W, D)  # [B, H*W, D]

        # Normalize features
        queries = F.normalize(queries, dim=2)
        keys = F.normalize(keys, dim=2)

        # Get positive keys using correspondence
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, H * W).to(correspondence.device)
        positive_keys = keys[batch_idx, correspondence]  # [B, H*W, D]

        # Compute positive similarities
        pos_sim = torch.sum(queries * positive_keys, dim=2) / self.temperature  # [B, H*W]

        # FIXED: Ensure queue is properly normalized and detached
        with torch.no_grad():
            queue_normalized = F.normalize(self.queue, dim=1)

        # Compute negative similarities with queue
        neg_sim = torch.mm(
            queries.view(-1, D),
            queue_normalized.t()
        ) / self.temperature  # [B*H*W, queue_size]

        # Compute loss for each spatial location
        logits = torch.cat([pos_sim.view(-1, 1), neg_sim], dim=1)  # [B*H*W, 1+queue_size]
        labels = torch.zeros(B * H * W, dtype=torch.long, device=queries.device)

        loss = F.cross_entropy(logits, labels)

        # FIXED: Update queue with keys from second view (with proper gradient blocking)
        # with torch.no_grad():
        #     keys_for_queue = keys.view(-1, D).clone()  # Clone to avoid gradient issues
        #     # Normalize keys before adding to queue
        #     keys_for_queue = F.normalize(keys_for_queue, dim=1)
        #     self._dequeue_and_enqueue(keys_for_queue)

        # Update queue with strategically sampled keys
        with torch.no_grad():
            # Sample diverse patches from keys
            sampled_keys, sampled_indices = self._sample_patches_strategically(
                keys, B, H, W
            )

            # Normalize sampled keys
            sampled_keys = F.normalize(sampled_keys, dim=2)

            # Update queue with diverse samples
            self._dequeue_and_enqueue_diverse(sampled_keys, B)

        return loss, pos_sim, neg_sim, queries, positive_keys, correspondence, queue_normalized