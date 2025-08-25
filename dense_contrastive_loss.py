import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseContrastiveLoss(nn.Module):
    """
    Dense Contrastive Loss as described in DenseCL paper
    """

    def __init__(self, temperature=0.2, queue_size=65536, momentum=0.999, correspondence_features='dense'):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum
        self.correspondence_features = correspondence_features

        # Initialize memory queue
        self.register_buffer("queue", torch.randn(queue_size, 128))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Normalize the queue
        self.queue = F.normalize(self.queue, dim=1)

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
        with torch.no_grad():
            keys_for_queue = keys.view(-1, D).clone()  # Clone to avoid gradient issues
            # Normalize keys before adding to queue
            keys_for_queue = F.normalize(keys_for_queue, dim=1)
            self._dequeue_and_enqueue(keys_for_queue)


        return loss, pos_sim, neg_sim