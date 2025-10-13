import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalContrastiveLoss(nn.Module):
    """
    Global Contrastive Loss (MoCo-style)
    Operates on global image representations (CLS token)
    """

    def __init__(self, temperature=0.2, queue_size=65536, momentum=0.999, learnable_temp=False, dense_dim=128):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.learnable_temp = learnable_temp

        # Initialize temperature
        if self.learnable_temp:
            init_log_temp = torch.log(torch.tensor(temperature))
            self.log_temperature = nn.Parameter(init_log_temp)
        else:
            self._fixed_temperature = temperature

        # Initialize memory queue
        self.register_buffer("queue", torch.randn(queue_size, dense_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Normalize the queue
        self.queue = F.normalize(self.queue, dim=1)

    @property
    def temperature(self):
        if self.learnable_temp:
            return torch.exp(self.log_temperature)
        else:
            return self._fixed_temperature

    def get_temperature_value(self):
        if self.learnable_temp:
            with torch.no_grad():
                return torch.exp(self.log_temperature).item()
        else:
            return self._fixed_temperature

    def _dequeue_and_enqueue(self, keys):
        """
        Update queue with global features from current batch
        Args:
            keys: [B, D] - Global features (CLS token projections)
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size].copy_(keys)
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Wrap around
            remaining_space = self.queue_size - ptr
            self.queue[ptr:].copy_(keys[:remaining_space])
            self.queue[:batch_size - remaining_space].copy_(keys[remaining_space:])
            ptr = batch_size - remaining_space

        self.queue_ptr[0] = ptr

    def forward(self, global_features_1, global_features_2):
        """
        Compute global contrastive loss
        Args:
            global_features_1: [B, D] - Projections from first view (CLS token)
            global_features_2: [B, D] - Projections from second view (CLS token)
        """
        B, D = global_features_1.shape

        # Normalize features
        q = F.normalize(global_features_1, dim=1)  # [B, D]
        k = F.normalize(global_features_2, dim=1)  # [B, D]

        # Positive pair similarity
        pos_sim = torch.sum(q * k, dim=1, keepdim=True) / self.temperature  # [B, 1]

        # Negative similarities with queue
        with torch.no_grad():
            queue_normalized = F.normalize(self.queue, dim=1)

        neg_sim = torch.mm(q, queue_normalized.t()) / self.temperature  # [B, queue_size]

        # Contrastive logits and loss
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1+queue_size]
        labels = torch.zeros(B, dtype=torch.long, device=q.device)

        loss = F.cross_entropy(logits, labels)

        # Update queue
        with torch.no_grad():
            self._dequeue_and_enqueue(k)

        return loss, pos_sim, neg_sim