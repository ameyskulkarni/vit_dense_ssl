import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import trunc_normal_
import math

class DenseContrastiveViT(nn.Module):
    """
    Vision Transformer with Dense Contrastive Learning
    Combines classification and dense contrastive learning objectives
    """

    def __init__(self,
                 model_name='vit_small_patch16_224',
                 num_classes=1000,
                 dense_dim=128,
                 pretrained=False,
                 drop_rate=0.0,
                 drop_path_rate=0.1
                 ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.dense_dim = dense_dim
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        self.vit, self.pretrained_checkpoint = self.build_vit()

        self.embed_dim = self.vit.embed_dim
        self.num_patches = self.vit.patch_embed.num_patches
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.grid_size = int(math.sqrt(self.num_patches))  # Assuming square patches

        # Classification head
        self.classification_head = nn.Linear(self.embed_dim, num_classes)

        # Dense projection head for contrastive learning
        # Dense Head v1.0
        # self.dense_projection_head = nn.Sequential(
        #     nn.Linear(self.embed_dim, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, dense_dim)
        # )

        # Dense Head v2.0
        self.dense_projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),  # ✅ Works with [B, N, D]
            nn.GELU(),  # Better activation for transformers
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, dense_dim)
        )

        # Initialize weights
        self.init_weights()

        print(f"------------------------------------------------------")
        print(f"Model params are: ")
        print(f"Model name: {self.model_name}")
        print(f"Pretrained: {self.pretrained}")
        print(f"------------------------------------------------------")

    def build_vit(self):
        if self.model_name == 'vit_tiny_patch16_224':
            return self.deit_tiny_patch16_224(self.pretrained)
        elif self.model_name == 'vit_small_patch16_224':
            return self.deit_small_patch16_224(self.pretrained)
        elif self.model_name == 'vit_base_patch16_224':
            return self.deit_base_patch16_224(self.pretrained)
        elif self.model_name == 'vit_very_tiny_patch16_224':
            return self.deit_very_tiny_patch16_224(self.pretrained)

    def deit_very_tiny_patch16_224(self, pretrained=False, **kwargs):
        print(f"Loading ViT very_tiny_patch16_224")
        model = VisionTransformer(
            patch_size=16, embed_dim=192, depth=1, num_heads=3, mlp_ratio=4, qkv_bias=True, drop_rate=self.drop_rate,drop_path_rate =self.drop_path_rate, num_classes=0, global_pool='token',
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        checkpoint = None
        # if pretrained:
        #     checkpoint = torch.hub.load_state_dict_from_url(
        #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #         map_location="cpu", check_hash=True
        #     )
            # model.load_state_dict(checkpoint["model"])
        return model, checkpoint

    def deit_tiny_patch16_224(self, pretrained=False, **kwargs):
        print(f"Loading ViT tiny_patch16_224")
        model = VisionTransformer(
            patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, drop_rate=self.drop_rate,drop_path_rate =self.drop_path_rate, num_classes=0, global_pool='token',
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        checkpoint = None
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
            # model.load_state_dict(checkpoint["model"])
        return model, checkpoint

    def deit_small_patch16_224(self, pretrained=False, **kwargs):
        print(f"Loading ViT small_patch16_224")
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=self.drop_rate,drop_path_rate =self.drop_path_rate, num_classes=0, global_pool='token',
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        checkpoint = None
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
            # model.load_state_dict(checkpoint["model"])
        return model, checkpoint


    def deit_base_patch16_224(self, pretrained=False, **kwargs):
        print(f"Loading ViT base_patch16_224")
        model = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_rate=self.drop_rate,drop_path_rate =self.drop_path_rate, num_classes=0, global_pool='token',
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        checkpoint = None
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            # model.load_state_dict(checkpoint["model"])
        return model, checkpoint

    def init_weights(self):
        """Initialize weights for the heads"""
        if self.pretrained:
            self.load_pretrained_backbone_weights()
            self.load_pretrained_classification_head()

        # Initialize dense projection head
        # for m in self.dense_projection_head:
        #     if isinstance(m, nn.Linear):
        #         trunc_normal_(m.weight, std=0.02)
        #         nn.init.constant_(m.bias, 0)

    def load_pretrained_backbone_weights(self):
        """Load pretrained ViT weights from timm, excluding the classification head"""
        print(f"Loading pretrained backbone weights for {self.model_name}...")

        try:
            # Load pretrained model with classification head
            pretrained_state_dict = self.pretrained_checkpoint["model"]

            # Get current model state dict
            current_state_dict = self.vit.state_dict()

            # Filter out classification head weights and load backbone weights
            filtered_state_dict = {}
            for key, value in pretrained_state_dict.items():
                # Skip head weights (classification head)
                if key.startswith('head'):
                    continue

                # Map the key if necessary and load if it exists in current model
                if key in current_state_dict:
                    if current_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        print(
                            f"Shape mismatch for {key}: current {current_state_dict[key].shape} vs pretrained {value.shape}")
                else:
                    print(f"Key {key} not found in current model")

            # Load the filtered state dict
            missing_keys, unexpected_keys = self.vit.load_state_dict(filtered_state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

            print(f"Successfully loaded pretrained weights for backbone!")

        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Falling back to random initialization...")

    def load_pretrained_classification_head(self, num_classes_pretrained=1000):
        """
        Load pretrained classification head weights if the number of classes matches
        """
        if self.classification_head.out_features != num_classes_pretrained:
            print(f"Cannot load pretrained classification head: "
                  f"current classes {self.classification_head.out_features} != pretrained classes {num_classes_pretrained}")
            return

        try:
            pretrained_state_dict = self.pretrained_checkpoint["model"]
            pretrained_head_weight = pretrained_state_dict["head.weight"].data
            pretrained_head_bias = pretrained_state_dict["head.bias"].data

            self.classification_head.weight.data.copy_(pretrained_head_weight)
            self.classification_head.bias.data.copy_(pretrained_head_bias)

            print("Successfully loaded pretrained classification head weights!")

        except Exception as e:
            print(f"Failed to load pretrained classification head: {e}")

    def forward_features(self, x):
        """Extract features from ViT backbone"""
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)

        if hasattr(self.vit, 'dist_token') and self.vit.dist_token is not None:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)

        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return x

    def forward(self, x, return_dense=True):
        """
        Forward pass
        Args:
            x: Input images [B, C, H, W]
            return_dense: Whether to return dense features for contrastive learning

        Returns:
            cls_output: Classification logits [B, num_classes]
            dense_features: Dense features [B, H, W, dense_dim] (if return_dense=True)
        """
        # Extract features
        features = self.forward_features(x)  # [B, 1+num_patches, embed_dim]

        # Split CLS token and patch tokens
        cls_token = features[:, 0]  # [B, embed_dim]
        patch_tokens = features[:, 1:]  # [B, num_patches, embed_dim]

        # Classification output
        cls_output = self.classification_head(cls_token)

        if not return_dense:
            return cls_output

        # Dense contrastive features
        dense_features = self.dense_projection_head(patch_tokens)  # [B, num_patches, dense_dim]

        # Reshape to spatial grid
        B, N, D = dense_features.shape
        H = W = self.grid_size
        dense_features = dense_features.reshape(B, H, W, D)
        backbone_features = patch_tokens.reshape(B, H, W, patch_tokens.shape[-1])

        return cls_output, dense_features, backbone_features

    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



def create_model(model_name='vit_tiny_patch16_224', num_classes=1000, pretrained=True, **kwargs):
    """Factory function to create DenseContrastiveViT model"""
    return DenseContrastiveViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Method 1: Create model with pretrained backbone
    print("=== Creating model with pretrained weights (backbone + classification_head ===")
    model = create_model('vit_tiny_patch16_224', num_classes=1000, pretrained=True)
    print(f"Model created with {model.get_num_params():,} parameters")

    # Test forward pass
    print("\n=== Testing forward pass ===")
    x = torch.randn(2, 3, 224, 224)
    cls_output, dense_features = model(x, return_dense=True)
    print(f"Classification output shape: {cls_output.shape}")
    print(f"Dense features shape: {dense_features.shape}")

    # Test evaluation mode (no dense features)
    model.eval()
    with torch.no_grad():
        cls_output_eval = model(x, return_dense=False)
        print(f"Evaluation output shape: {cls_output_eval.shape}")