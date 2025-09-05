import torch
from collections import defaultdict

class WeightTracker:
    def __init__(self):
        self.previous_weights = {}
        self.weight_changes = defaultdict(list)
        self.iteration_count = 0

    def _store_current_weights(self, model):
        """Store current model weights"""
        self.previous_weights = {}
        for name, param in model.dense_projection_head.named_parameters():
            if param.requires_grad:
                self.previous_weights[name] = param.data.clone().detach()

    def track_weight_changes(self, model):
        """Track changes in model weights since last call"""
        changes = {}

        for name, param in model.dense_projection_head.named_parameters():
            if param.requires_grad and name in self.previous_weights:
                # Calculate weight change
                change = torch.norm(param.data - self.previous_weights[name])
                changes[f'weight_changes/dense_head_{name}'] = change

        # Update stored weights for next iteration
        self._store_current_weights(model)

        return changes