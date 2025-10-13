import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import logging



class ModelEvaluator:
    """
    Comprehensive evaluation pipeline for Dense Contrastive ViT
    """

    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.load_model()
        self.setup_data()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.get('save_dir', './results'))
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_device(self):
        """Setup device for evaluation"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the trained model"""
        # Create model
        self.model = ContrastiveViT(
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes'],
            dense_dim=self.config.get('dense_dim', 128),
            pretrained=self.config.get('pretrained', False)
        ).to(self.device)

        # Load checkpoint
        if self.config.get('checkpoint_path'):
            checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device)

            # Handle DataParallel models
            state_dict = checkpoint['state_dict']
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)
            self.logger.info(f"Loaded checkpoint from {self.config['checkpoint_path']}")

            # Log training metrics if available
            if 'metrics' in checkpoint:
                self.logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")

        self.model.eval()
        self.logger.info(f"Model loaded with {self.model.get_num_params():,} parameters")

    def setup_data(self):
        """Setup data loaders for evaluation"""
        # Standard ImageNet evaluation transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        self.datasets = {}
        self.data_loaders = {}

        # ImageNet validation set
        if self.config.get('imagenet_val_path'):
            self.datasets['imagenet_val'] = ImageFolder(
                root=self.config['imagenet_val_path'],
                transform=self.transform
            )
            self.data_loaders['imagenet_val'] = DataLoader(
                self.datasets['imagenet_val'],
                batch_size=self.config.get('batch_size', 64),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )

        # Additional test datasets (if provided)
        for dataset_name, dataset_path in self.config.get('additional_datasets', {}).items():
            if os.path.exists(dataset_path):
                self.datasets[dataset_name] = ImageFolder(
                    root=dataset_path,
                    transform=self.transform
                )
                self.data_loaders[dataset_name] = DataLoader(
                    self.datasets[dataset_name],
                    batch_size=self.config.get('batch_size', 64),
                    shuffle=False,
                    num_workers=self.config.get('num_workers', 4),
                    pin_memory=True
                )
                self.logger.info(f"Added dataset {dataset_name} with {len(self.datasets[dataset_name])} samples")

    def evaluate_accuracy(self, dataset_name: str) -> Dict[str, float]:
        """Evaluate standard accuracy metrics"""
        if dataset_name not in self.data_loaders:
            raise ValueError(f"Dataset {dataset_name} not found")

        data_loader = self.data_loaders[dataset_name]

        # Metrics tracking
        correct_1 = 0
        correct_5 = 0
        total = 0
        all_predictions = []
        all_targets = []

        self.logger.info(f"Evaluating accuracy on {dataset_name}")

        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc=f'Evaluating {dataset_name}'):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass (no dense features during evaluation)
                outputs = self.model(images, return_dense=False)

                # Top-1 and Top-5 accuracy
                _, pred_1 = outputs.topk(1, 1, True, True)
                _, pred_5 = outputs.topk(5, 1, True, True)

                pred_1 = pred_1.t()
                pred_5 = pred_5.t()

                correct_1 += pred_1.eq(targets.view(1, -1).expand_as(pred_1)).sum().item()
                correct_5 += pred_5.eq(targets.view(1, -1).expand_as(pred_5)).sum().item()

                total += targets.size(0)

                # Store predictions and targets for detailed analysis
                all_predictions.extend(pred_1.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        acc_1 = 100.0 * correct_1 / total
        acc_5 = 100.0 * correct_5 / total

        results = {
            'top1_accuracy': acc_1,
            'top5_accuracy': acc_5,
            'total_samples': total,
            'predictions': all_predictions,
            'targets': all_targets
        }

        self.logger.info(f"{dataset_name} Results:")
        self.logger.info(f"  Top-1 Accuracy: {acc_1:.2f}%")
        self.logger.info(f"  Top-5 Accuracy: {acc_5:.2f}%")
        self.logger.info(f"  Total Samples: {total}")

        return results

    def evaluate_per_class_accuracy(self, dataset_name: str) -> Dict[str, float]:
        """Evaluate per-class accuracy"""
        if dataset_name not in self.data_loaders:
            raise ValueError(f"Dataset {dataset_name} not found")

        data_loader = self.data_loaders[dataset_name]
        num_classes = len(self.datasets[dataset_name].classes)

        # Per-class metrics
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)

        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc=f'Per-class eval {dataset_name}'):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(images, return_dense=False)
                _, predictions = torch.max(outputs, 1)

                # Update per-class counters
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predictions[i] == label:
                        class_correct[label] += 1

        # Calculate per-class accuracy
        per_class_acc = {}
        class_names = self.datasets[dataset_name].classes

        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100.0 * class_correct[i] / class_total[i]
                per_class_acc[class_names[i]] = accuracy
            else:
                per_class_acc[class_names[i]] = 0.0

        # Calculate statistics
        accuracies = list(per_class_acc.values())
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)

        self.logger.info(f"{dataset_name} Per-class Results:")
        self.logger.info(f"  Mean Accuracy: {mean_acc:.2f}% (±{std_acc:.2f})")
        self.logger.info(f"  Min Accuracy: {min_acc:.2f}%")
        self.logger.info(f"  Max Accuracy: {max_acc:.2f}%")

        return {
            'per_class_accuracy': per_class_acc,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc
        }

    def evaluate_inference_speed(self, dataset_name: str = None, num_samples: int = 1000) -> Dict[str, float]:
        """Evaluate inference speed"""
        if dataset_name and dataset_name in self.data_loaders:
            data_loader = self.data_loaders[dataset_name]
        else:
            # Create dummy data if no dataset provided
            dummy_data = torch.randn(1, 3, 224, 224).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if dataset_name:
                    images, _ = next(iter(data_loader))
                    images = images[:1].to(self.device)
                else:
                    images = dummy_data
                _ = self.model(images, return_dense=False)

        # Measure inference time
        times = []
        samples_processed = 0

        with torch.no_grad():
            if dataset_name:
                for images, _ in data_loader:
                    if samples_processed >= num_samples:
                        break

                    images = images.to(self.device, non_blocking=True)

                    start_time = time.time()
                    _ = self.model(images, return_dense=False)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()

                    batch_time = end_time - start_time
                    times.append(batch_time / images.size(0))  # Per-sample time
                    samples_processed += images.size(0)
            else:
                for _ in range(num_samples):
                    start_time = time.time()
                    _ = self.model(dummy_data, return_dense=False)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()

                    times.append(end_time - start_time)

        # Calculate statistics
        mean_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        throughput = 1.0 / np.mean(times)  # Images per second

        results = {
            'mean_inference_time_ms': mean_time,
            'std_inference_time_ms': std_time,
            'throughput_fps': throughput,
            'samples_evaluated': len(times)
        }

        self.logger.info(f"Inference Speed Results:")
        self.logger.info(f"  Mean Time: {mean_time:.2f}ms (±{std_time:.2f})")
        self.logger.info(f"  Throughput: {throughput:.1f} FPS")

        return results

    def generate_confusion_matrix(self, dataset_name: str, save_path: Optional[str] = None) -> np.ndarray:
        """Generate and save confusion matrix"""
        if dataset_name not in self.data_loaders:
            raise ValueError(f"Dataset {dataset_name} not found")

        # Get predictions
        results = self.evaluate_accuracy(dataset_name)
        predictions = results['predictions']
        targets = results['targets']

        # Generate confusion matrix
        cm = confusion_matrix(targets, predictions)

        # Plot confusion matrix (for smaller number of classes)
        if len(self.datasets[dataset_name].classes) <= 50:
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {dataset_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Confusion matrix saved to {save_path}")

            plt.close()

        return cm

    def evaluate_model_size(self) -> Dict[str, float]:
        """Evaluate model size and memory usage"""
        # Calculate model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)

        # Memory usage during inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input, return_dense=False)

            memory_usage_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            memory_usage_mb = 0.0

        results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'memory_usage_mb': memory_usage_mb
        }

        self.logger.info(f"Model Size Results:")
        self.logger.info(f"  Total Parameters: {total_params:,}")
        self.logger.info(f"  Trainable Parameters: {trainable_params:,}")
        self.logger.info(f"  Model Size: {model_size_mb:.2f} MB")
        self.logger.info(f"  Memory Usage: {memory_usage_mb:.2f} MB")

        return results

    def run_comprehensive_evaluation(self) -> Dict[str, any]:
        """Run comprehensive evaluation on all datasets"""
        results = {}
        per_class_results = {}

        # Model size evaluation
        results['model_info'] = self.evaluate_model_size()

        # Inference speed evaluation
        results['inference_speed'] = self.evaluate_inference_speed()

        # Evaluate on all datasets
        for dataset_name in self.data_loaders.keys():
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Evaluating on {dataset_name}")
            self.logger.info(f"{'=' * 50}")

            # Accuracy evaluation
            acc_results = self.evaluate_accuracy(dataset_name)

            # Per-class accuracy
            # per_class_results = self.evaluate_per_class_accuracy(dataset_name)

            # Inference speed on this dataset
            speed_results = self.evaluate_inference_speed(dataset_name)

            # Generate confusion matrix for smaller datasets
            # if len(self.datasets[dataset_name].classes) <= 50:
            #     cm_path = Path(self.config.get('save_dir', './results')) / f'confusion_matrix_{dataset_name}.png'
            #     self.generate_confusion_matrix(dataset_name, str(cm_path))

            results[dataset_name] = {
                'accuracy': acc_results,
                'per_class': per_class_results,
                'speed': speed_results
            }

        # Save results
        self.save_results(results)

        return results

    def save_results(self, results: Dict[str, any]):
        """Save evaluation results"""
        save_dir = Path(self.config.get('save_dir', './results'))
        save_dir.mkdir(parents=True, exist_ok=True)

        # Remove non-serializable items
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        serializable_results[key][sub_key] = {
                            k: v for k, v in sub_value.items()
                            if not isinstance(v, (np.ndarray, list)) or k in ['per_class_accuracy']
                        }
                    else:
                        serializable_results[key][sub_key] = sub_value
            else:
                serializable_results[key] = value

        # Save as JSON
        results_path = save_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {results_path}")

        # Create summary report
        self.create_summary_report(serializable_results, save_dir / 'summary_report.txt')

    def create_summary_report(self, results: Dict[str, any], save_path: Path):
        """Create a human-readable summary report"""
        with open(save_path, 'w') as f:
            f.write("Dense Contrastive Vision Transformer - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            # Model information
            if 'model_info' in results:
                f.write("MODEL INFORMATION:\n")
                f.write("-" * 20 + "\n")
                info = results['model_info']
                f.write(f"Total Parameters: {info['total_parameters']:,}\n")
                f.write(f"Model Size: {info['model_size_mb']:.2f} MB\n")
                f.write(f"Memory Usage: {info['memory_usage_mb']:.2f} MB\n\n")

            # Inference speed
            if 'inference_speed' in results:
                f.write("INFERENCE SPEED:\n")
                f.write("-" * 20 + "\n")
                speed = results['inference_speed']
                f.write(f"Mean Time: {speed['mean_inference_time_ms']:.2f}ms\n")
                f.write(f"Throughput: {speed['throughput_fps']:.1f} FPS\n\n")

            # Dataset results
            for dataset_name, dataset_results in results.items():
                if dataset_name in ['model_info', 'inference_speed']:
                    continue

                f.write(f"DATASET: {dataset_name.upper()}\n")
                f.write("-" * 30 + "\n")

                if 'accuracy' in dataset_results:
                    acc = dataset_results['accuracy']
                    f.write(f"Top-1 Accuracy: {acc['top1_accuracy']:.2f}%\n")
                    f.write(f"Top-5 Accuracy: {acc['top5_accuracy']:.2f}%\n")
                    f.write(f"Total Samples: {acc['total_samples']}\n")

                # if 'per_class' in dataset_results:
                #     per_class = dataset_results['per_class']
                #     f.write(f"Mean Class Accuracy: {per_class['mean_accuracy']:.2f}%\n")
                #     f.write(f"Std Class Accuracy: {per_class['std_accuracy']:.2f}%\n")

                f.write("\n")

        self.logger.info(f"Summary report saved to {save_path}")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate Dense Contrastive ViT')
    parser.add_argument('--checkpoint', type=str, required=False, default=None, help='Path to model checkpoint')
    parser.add_argument('--imagenet-val', type=str, default='/home/cognition/datasets/IMAGENET/imagenet_ILSVRC-2012_ImageNet-1K/val', help='Path to ImageNet validation set')
    parser.add_argument('--model-name', type=str, default='vit_tiny_patch16_224', help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save-dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--pretrained', type=bool, required=False, default=True, help='If to use deit pretrained weights')

    args = parser.parse_args()

    # Configuration
    config = {
        'checkpoint_path': args.checkpoint,
        'imagenet_val_path': args.imagenet_val,
        'model_name': args.model_name,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'save_dir': args.save_dir,
        'additional_datasets': {
            # Add additional test datasets here
            # 'dataset_name': 'path/to/dataset'
        },
        'dense_dim': 128,
        'pretrained': args.pretrained,
    }
    print(config)

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(config)
    results = evaluator.run_comprehensive_evaluation()

    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {config['save_dir']}")


if __name__ == "__main__":
    main()