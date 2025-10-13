import torch
import argparse
import os
from collections import OrderedDict


def convert_contrastive_vit_to_deit(checkpoint_path, output_path, model_name='vit_tiny_patch16_224'):
    """
    Convert ContrastiveViT checkpoint to standard DeiT checkpoint format

    Args:
        checkpoint_path: Path to the ContrastiveViT checkpoint
        output_path: Path where to save the converted DeiT checkpoint
        model_name: Model architecture name
    """

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract the model state dict
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    else:
        model_state_dict = checkpoint

    print(f"Original checkpoint keys: {len(model_state_dict)}")

    # Create new state dict for DeiT format
    deit_state_dict = OrderedDict()

    # Mapping from ContrastiveViT keys to DeiT keys
    key_mapping = {
        # ViT backbone mappings
        'vit.patch_embed.proj.weight': 'patch_embed.proj.weight',
        'vit.patch_embed.proj.bias': 'patch_embed.proj.bias',
        'vit.cls_token': 'cls_token',
        'vit.dist_token': 'dist_token',  # if exists
        'vit.pos_embed': 'pos_embed',
        'vit.pos_drop': 'pos_drop',
        'vit.norm.weight': 'norm.weight',
        'vit.norm.bias': 'norm.bias',

        # Classification head mapping
        'classification_head.weight': 'head.weight',
        'classification_head.bias': 'head.bias',
    }

    # Process the state dict
    for old_key, tensor in model_state_dict.items():
        new_key = None

        # Handle ViT blocks (transformer layers)
        if old_key.startswith('vit.blocks.'):
            # Remove 'vit.' prefix from blocks
            new_key = old_key.replace('vit.blocks.', 'blocks.')

        # Handle direct mappings
        elif old_key in key_mapping:
            new_key = key_mapping[old_key]

        # Handle other vit.* keys by removing the 'vit.' prefix
        elif old_key.startswith('vit.'):
            new_key = old_key.replace('vit.', '')

        # Skip contrastive projection head and other custom components
        elif old_key.startswith('dense_projection_head.'):
            print(f"Skipping dense projection head key: {old_key}")
            continue
        elif old_key.startswith('global_projection_head.'):
            print(f"Skipping global projection head key: {old_key}")
            continue

        # Skip any other custom keys
        else:
            print(f"Skipping unknown key: {old_key}")
            continue

        if new_key:
            deit_state_dict[new_key] = tensor
            print(f"Mapped: {old_key} -> {new_key}")

    print(f"Converted checkpoint keys: {len(deit_state_dict)}")

    # Create the final checkpoint in DeiT format
    final_checkpoint = {
        'model': deit_state_dict,
        'epoch': checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0,
        'arch': model_name,
    }

    # Copy other metadata if available
    if isinstance(checkpoint, dict):
        for key in ['optimizer', 'lr_scheduler', 'scaler', 'args']:
            if key in checkpoint:
                final_checkpoint[key] = checkpoint[key]

    # Save the converted checkpoint
    print(f"Saving converted checkpoint to: {output_path}")
    torch.save(final_checkpoint, output_path)

    print("Conversion completed successfully!")

    # Print summary
    print(f"\nSummary:")
    print(f"- Input checkpoint: {checkpoint_path}")
    print(f"- Output checkpoint: {output_path}")
    print(f"- Original keys: {len(model_state_dict) if 'model_state_dict' in locals() else 'N/A'}")
    print(f"- Converted keys: {len(deit_state_dict)}")

    return final_checkpoint


def verify_checkpoint_compatibility(checkpoint_path, model_name='vit_tiny_patch16_224'):
    """
    Verify that the converted checkpoint can be loaded by a standard DeiT model
    """
    try:
        import timm

        print(f"\nVerifying checkpoint compatibility...")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # Create a timm model for verification
        if model_name == 'vit_tiny_patch16_224':
            model = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=1000)
        elif model_name == 'vit_small_patch16_224':
            model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=1000)
        elif model_name == 'vit_base_patch16_224':
            model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=1000)
        else:
            print(f"Unknown model name: {model_name}")
            return False

        # Try to load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        if not missing_keys and not unexpected_keys:
            print("✅ Checkpoint is fully compatible!")
        else:
            print("⚠️  Checkpoint has some compatibility issues but may still work")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")

        return True

    except ImportError:
        print("timm not available for verification. Install with: pip install timm")
        return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert ContrastiveViT checkpoint to DeiT format')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input ContrastiveViT checkpoint')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output DeiT checkpoint')
    parser.add_argument('--model_name', '-m', type=str, default='vit_tiny_patch16_224',
                        choices=['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224'],
                        help='Model architecture name')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the converted checkpoint compatibility')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input checkpoint file not found: {args.input}")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert the checkpoint
    try:
        convert_contrastive_vit_to_deit(args.input, args.output, args.model_name)

        # Verify if requested
        if args.verify:
            verify_checkpoint_compatibility(args.output, args.model_name)

    except Exception as e:
        print(f"Error during conversion: {e}")


# Example usage
if __name__ == "__main__":
    # If running as script
    import sys

    if len(sys.argv) == 1:
        # Example usage for demonstration
        print("Example usage:")
        print(
            "python contrastive_vit_to_deit_format.py -i model_checkpoint.pth -o deit_checkpoint.pth -m vit_tiny_patch16_224 --verify")
        print("\nYou can also use it programmatically:")

        # Programmatic example (uncomment to use)
        # convert_contrastive_vit_to_deit(
        #     checkpoint_path='path/to/your/contrastive_vit_checkpoint.pth',
        #     output_path='path/to/output/deit_checkpoint.pth',
        #     model_name='vit_tiny_patch16_224'
        # )
    else:
        main()