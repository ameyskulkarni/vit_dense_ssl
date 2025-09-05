import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from datasets import load_dataset
import os
import time
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, Optional

from dense_contrastive_vit_model import DenseContrastiveViT
from dense_contrastive_loss import DenseContrastiveLoss
from contrastive_image_dataset import ContrastiveImageDataset
from sketch_imagenet_dataset_builder import ImageNetSketchDataset
from dataset_class_matching import ImageNetV2Dataset, ImageNetADataset, ImageNetRDataset
from compute_similarity_stats import ContrastiveLearningMetrics
from compute_feature_collapse import compute_feature_rank
from weight_tracker import WeightTracker


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class DenseContrastiveTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_criterion()
        # Initialize wandb if enabled
        self.init_wandb()

    def init_wandb(self):
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'dense-contrastive-vit'),
                config=self.config,
                name=self.config.get('experiment_name', 'dense_vit_tiny_baseline')
            )
            # self.wandb_log_data = {}

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['save_dir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_device(self):
        """Setup device for training"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        else:
            self.logger.info("Using CPU")

    def setup_model(self):
        """Setup model architecture"""
        self.model = DenseContrastiveViT(
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes'],
            dense_dim=self.config['dense_dim'],
            drop_rate=self.config.get('drop_rate', 0.0),
            drop_path_rate=self.config.get('drop_path_rate', 0.1),
            pretrained=self.config.get('pretrained', False),
        ).to(self.device)

        self.logger.info(f"Model created with {self.model.get_num_params():,} parameters")

        # Enable DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1 and self.config.get('model_parallel', False):
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")

    def setup_data(self):
        """Setup data loaders"""
        # Data augmentation for training
        train_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Remove RandomApply wrapper
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
            ], p=0.2),  # Reduce both kernel size and probability
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Validation transform
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create contrastive training dataset
        train_dataset = ContrastiveImageDataset(
            root=os.path.join(self.config['data_path'], 'train'),
            transform1=train_transform1,
            transform2=train_transform2
        )

        # Imagenet standard val dataloader
        val_dataset = ImageFolder(
            root=os.path.join(self.config['data_path'], 'val'),
            transform=val_transform
        )
        val_data_class_name_idx_mapping = val_dataset.class_to_idx

        # Write to file
        # with open('/home/cognition/projects/vit_dense_ssl/imagenet_to_imagefolder.txt', 'w') as f:
        #     for class_name, class_idx in val_dataset.class_to_idx.items():
        #         f.write(f"{class_idx}\t{class_name}\n")

        # Stylized Imagenet val dataloader
        stylized_imagenet_val_dataset = ImageFolder(
            root=os.path.join(self.config['stylized_imagenet'], 'val'),
            transform=val_transform
        )

        # Imagenet-sketch val dataset
        sketch_IN_dataset = load_dataset("imagenet_sketch", split="train")
        # Wrap it with the custom dataset
        sketch_imagenet_val_dataset = ImageNetSketchDataset(sketch_IN_dataset, transform=val_transform)

        # Imagenet A val dataloader
        imagenet_A_val_dataset = ImageNetADataset(
            root=os.path.join(self.config['imagenet_A']),
            mapping=val_data_class_name_idx_mapping,
            transform=val_transform
        )

        # Imagenet R val dataloader
        imagenet_R_val_dataset = ImageNetRDataset(
            root=os.path.join(self.config['imagenet_R']),
            mapping=val_data_class_name_idx_mapping,
            transform=val_transform
        )

        # Imagenet v2 val datasets
        # ImageNet-v2 variants (distribution shift robustness)
        imagenet_v2_matched_freq_dataset = ImageNetV2Dataset(
            root=os.path.join(self.config['imagenet_v2'], 'imagenetv2-matched-frequency'),
            transform=val_transform
        )

        imagenet_v2_threshold_dataset = ImageNetV2Dataset(
            root=os.path.join(self.config['imagenet_v2'], 'imagenetv2-threshold0.7'),
            transform=val_transform
        )

        imagenet_v2_top_images_dataset = ImageNetV2Dataset(
            root=os.path.join(self.config['imagenet_v2'], 'imagenetv2-top-images'),
            transform=val_transform
        )


        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True
        )

        # Imagenet standard val dataloader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size= 3 * self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # Stylized Imagenet val dataloader
        self.stylized_imagenet_val_loader = DataLoader(
            stylized_imagenet_val_dataset,
            batch_size= 3 * self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # Sketch Imagenet val dataloader
        self.sketch_imagenet_val_loader = DataLoader(
            sketch_imagenet_val_dataset,
            batch_size= 3 * self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # Stylized Imagenet val dataloader
        self.imagenet_A_val_loader = DataLoader(
            imagenet_A_val_dataset,
            batch_size= 3 * self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # Stylized Imagenet val dataloader
        self.imagenet_R_val_loader = DataLoader(
            imagenet_R_val_dataset,
            batch_size= 3 * self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # ImageNet-V2 variant loaders
        self.imagenet_v2_matched_freq_loader = DataLoader(
            imagenet_v2_matched_freq_dataset,
            batch_size= 3 * self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.imagenet_v2_threshold_loader = DataLoader(
            imagenet_v2_threshold_dataset,
            batch_size= 3 * self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.imagenet_v2_top_images_loader = DataLoader(
            imagenet_v2_top_images_dataset,
            batch_size= 3 * self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        self.logger.info(f"Stylized Imagenet Validation dataset size: {len(stylized_imagenet_val_dataset)}")
        self.logger.info(f"Sketch Imagenet Validation dataset size: {len(sketch_imagenet_val_dataset)}")
        self.logger.info(f"Imagenet_A Validation dataset size: {len(imagenet_A_val_dataset)}")
        self.logger.info(f"Imagenet_R Validation dataset size: {len(imagenet_R_val_dataset)}")
        self.logger.info(f"Imagenet_v2_matched_freq dataset size: {len(imagenet_v2_matched_freq_dataset)}")
        self.logger.info(f"Imagenet_v2_threshold dataset size: {len(imagenet_v2_threshold_dataset)}")
        self.logger.info(f"Imagenet_v2_top_images dataset size: {len(imagenet_v2_top_images_dataset)}")

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )

        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )

        # Warmup scheduler
        self.warmup_epochs = self.config.get('warmup_epochs', 5)
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )

    def setup_criterion(self):
        """Setup loss functions"""
        # Classification loss
        self.classification_criterion = nn.CrossEntropyLoss()

        # Dense contrastive loss
        self.dense_contrastive_criterion = DenseContrastiveLoss(
            temperature=self.config.get('temperature', 0.2),
            queue_size=self.config.get('queue_size', 65536),
            momentum=self.config.get('momentum', 0.999),
            correspondence_features=self.config.get('correspondence_features', 'dense'),
            max_patches_per_image=50,  # Sample 50 patches per image
            sampling_strategy='random'  # or 'random', 'hardest'
        ).to(self.device)

        # Loss weight
        self.lambda_weight = self.config.get('lambda_weight', 0.5)

    def get_global_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5


    def train_epoch(self, epoch: int, weight_tracker: WeightTracker) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()

        # Metrics
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        cls_losses = AverageMeter('ClsLoss', ':.4e')
        dense_losses = AverageMeter('DenseLoss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        cl_metrics = ContrastiveLearningMetrics()

        end = time.time()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for i, (images, targets) in enumerate(pbar):
            training_analysis_metric = {}
            # Measure data loading time
            data_time.update(time.time() - end)

            images_1, images_2 = images[0], images[1]
            images_1 = images_1.to(self.device, non_blocking=True)
            images_2 = images_2.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            if self.lambda_weight > 0:

                cls_output_1, dense_features_1, backbone_features_1 = self.model(images_1, return_dense=True)
                cls_output_2, dense_features_2, backbone_features_2 = self.model(images_2, return_dense=True)
                # cls_output_1.shape: [64, 1000], dense_features_1.shape: [64, 14, 14, 128]
                # cls_output_2.shape: [64, 1000], dense_features_2.shape: [64, 14, 14, 128]

                # Dense contrastive loss
                # For correspondence, we use the dense features as backbone features
                # In practice, you might want to use features from an earlier layer
                dense_loss, pos_sim, neg_sim, queries, positive_keys, correspondence, neg_queue_features = self.dense_contrastive_criterion(
                    dense_features_1, dense_features_2,
                    backbone_features_1, backbone_features_2  # Using same features for correspondence
                )
                if self.config.get('correspondence_features', 'dense'):
                    corr_features_1 = dense_features_1
                    corr_features_2 = dense_features_2
                else:
                    corr_features_1 = backbone_features_1
                    corr_features_2 = backbone_features_2

                if i % 10 == 0:
                    sim_stats = cl_metrics.compute_metrics(queries, positive_keys, neg_sim, pos_sim, correspondence,
                        corr_features_1, corr_features_2, neg_queue_features)
                    effective_rank1, eigenvals1 = compute_feature_rank(dense_features_1)
                    effective_rank2, eigenvals2 = compute_feature_rank(dense_features_2)
                    weight_changes = weight_tracker.track_weight_changes(self.model)
                    diversity_stats = self.dense_contrastive_criterion.get_queue_diversity_stats()
                    training_analysis_metric.update(sim_stats)
                    training_analysis_metric.update(weight_changes)
                    training_analysis_metric['feature_collapse/effective_rank1'] = effective_rank1
                    training_analysis_metric['feature_collapse/eigenvals1'] = eigenvals1
                    training_analysis_metric['feature_collapse/effective_rank2'] = effective_rank2
                    training_analysis_metric['feature_collapse/eigenvals2'] = eigenvals2
                    training_analysis_metric['feature_diversity_stats/unique_images_in_queue'] = diversity_stats['unique_images_in_queue']
            else:
                # Skip dense computations entirely
                cls_output_1 = self.model(images_1, return_dense=False)  # or just self.model(images_1)
                cls_output_2 = self.model(images_2, return_dense=False)
                dense_loss = 0

            # Classification loss (using both views)
            cls_loss = (self.classification_criterion(cls_output_1, targets) +
                        self.classification_criterion(cls_output_2, targets)) / 2



            # Total loss
            if self.config.get('contrastive_weight_adaptive', False):
                # Start with classification-focused training
                warmup_target = 0.2 * self.lambda_weight
                if epoch < self.warmup_epochs:
                    adaptive_lambda = warmup_target * ((epoch + 1) / self.warmup_epochs)  # First warmup epochs: 0 -> 20% of self.lambda (upperbound)
                else:
                    # Then gradually increase
                    progress = (epoch - self.warmup_epochs + 1) / (self.config.get('epochs', 55) - self.warmup_epochs)
                    adaptive_lambda = warmup_target + progress * (self.lambda_weight - warmup_target)
                
                total_loss = (1 - adaptive_lambda) * cls_loss + adaptive_lambda * dense_loss
            else:
                total_loss = (1 - self.lambda_weight) * cls_loss + self.lambda_weight * dense_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Logging dense head gradients.
            if i % 10 == 0:
                for name, param in self.model.dense_projection_head.named_parameters():
                    if param.grad is not None:
                        training_analysis_metric[f'gradients/before_clip_dense_head_{name}_norm'] = param.grad.detach().data.norm(2)
                        training_analysis_metric[f'gradients/before_clip_dense_head_{name}_mean'] = param.grad.detach().data.mean()
                        training_analysis_metric[f'gradients/before_clip_dense_head_{name}_std'] = param.grad.detach().data.std()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                grad_norm = self.get_global_grad_norm()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            # Logging dense head gradients after gradient clipping.
            if i % 10 == 0:
                for name, param in self.model.dense_projection_head.named_parameters():
                    if param.grad is not None:
                        training_analysis_metric[f'gradients/after_clip_dense_head_{name}_norm'] = param.grad.detach().data.norm(2)
                        training_analysis_metric[f'gradients/after_clip_dense_head_{name}_mean'] = param.grad.detach().data.mean()
                        training_analysis_metric[f'gradients/after_clip_dense_head_{name}_std'] = param.grad.detach().data.std()

            self.optimizer.step()

            # Measure accuracy
            acc1, acc5 = self.accuracy(cls_output_1, targets, topk=(1, 5))

            # Update metrics
            losses.update(total_loss.item(), images[0].size(0))
            cls_losses.update(cls_loss.item(), images[0].size(0))
            if self.lambda_weight > 0:
                dense_losses.update(dense_loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))
            top5.update(acc5[0], images[0].size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'ClsLoss': f'{cls_losses.avg:.4f}',
                'DenseLoss': f'{dense_losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Log to wandb
            if self.config.get('use_wandb', False) and i % self.config.get('log_interval', 100) == 0:
                wandb.log({
                    'train/loss': losses.avg,
                    'train/cls_loss': cls_losses.avg,
                    'train/dense_loss': dense_losses.avg,
                    'train/acc1': top1.avg,
                    'train/acc5': top5.avg,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/grad_norm': grad_norm,
                    'epoch': epoch
                })
                if i % 10 == 0:
                    wandb.log(training_analysis_metric)

        return {
            'loss': losses.avg,
            'cls_loss': cls_losses.avg,
            'dense_loss': dense_losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg
        }

    def validate(self, epoch: int, data_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        with torch.no_grad():
            pbar = tqdm(data_loader, desc='Validation')

            for i, (images, targets) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass (no dense features for validation)
                cls_output = self.model(images, return_dense=False)
                # print(f"cls_output: {cls_output}/ targets: {targets}")

                # Classification loss
                loss = self.classification_criterion(cls_output, targets)

                # Measure accuracy
                acc1, acc5 = self.accuracy(cls_output, targets, topk=(1, 5))

                # Update metrics
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc@1': f'{top1.avg:.2f}',
                    'Acc@5': f'{top5.avg:.2f}'
                })

        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg
        }

    def validate_imagenet_c_by_corruption(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """
        Validate on ImageNet-C broken down by corruption type and severity
        Requires ImageNet-C to be organized as: corruption_type/severity_level/class_folders/
        """
        corruption_results = {}

        corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
            'elastic_transform', 'pixelate', 'jpeg_compression'
        ]

        for corruption in corruption_types:
            corruption_results[corruption] = {}

            for severity in range(1, 6):  # ImageNet-C has 5 severity levels
                corruption_path = os.path.join(
                    self.config['imagenet_C'],
                    corruption,
                    str(severity)
                )

                if os.path.exists(corruption_path):
                    # Create dataset for this specific corruption and severity
                    corruption_dataset = ImageFolder(
                        root=corruption_path,
                        transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                    )

                    corruption_loader = DataLoader(
                        corruption_dataset,
                        batch_size= 3 * self.config['batch_size'],
                        shuffle=False,
                        num_workers=self.config['num_workers'],
                        pin_memory=True
                    )

                    # Validate on this corruption/severity combination
                    result = self.validate(epoch, corruption_loader)
                    corruption_results[corruption][f'severity_{severity}'] = result

        return corruption_results

    def calculate_imagenet_c_metrics(self, corruption_results: Dict) -> Dict[str, float]:
        """
        Calculate standard ImageNet-C metrics: mCE (mean Corruption Error) and relative mCE
        """
        # AlexNet error rates for each corruption type (used as baseline for mCE calculation)
        alexnet_err = {
            'gaussian_noise': 0.886, 'shot_noise': 0.894, 'impulse_noise': 0.922,
            'defocus_blur': 0.819, 'glass_blur': 0.826, 'motion_blur': 0.785,
            'zoom_blur': 0.798, 'snow': 0.866, 'frost': 0.826, 'fog': 0.819,
            'brightness': 0.564, 'contrast': 0.853, 'elastic_transform': 0.646,
            'pixelate': 0.717, 'jpeg_compression': 0.606
        }

        ce_scores = []
        average_ce_per_corruption = {}

        for corruption_type, severities in corruption_results.items():
            if corruption_type in alexnet_err:
                # Calculate average error across all severities for this corruption
                corruption_errors = []
                for severity_key, result in severities.items():
                    error_rate = 1.0 - (result['acc1'] / 100.0)  # Convert accuracy to error rate
                    corruption_errors.append(error_rate)

                if corruption_errors:
                    avg_error = sum(corruption_errors) / len(corruption_errors)
                    average_ce_per_corruption[corruption_type] = avg_error
                    ce_score = avg_error / alexnet_err[corruption_type]
                    ce_scores.append(ce_score)

        mce = sum(ce_scores) / len(ce_scores) if ce_scores else 0.0

        return {
            'average_ce_per_corruption': average_ce_per_corruption,
            'mCE': mce,
            'mCE_percent': mce * 100
        }

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = Path(self.config['save_dir']) / f'checkpoint_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = Path(self.config['save_dir']) / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best checkpoint saved at epoch {epoch}")

    def train(self):
        """Main training loop"""
        best_acc1 = 0.0
        weight_tracker = WeightTracker()
        weight_tracker._store_current_weights(self.model)

        for epoch in range(1, self.config['epochs'] + 1):
            self.logger.info(f"Epoch {epoch}/{self.config['epochs']}")


            if epoch == 1 and not self.config.get('no_log_baseline_res', True):
                val_metrics = self.validate(0, self.val_loader)
                stylized_imagenet_val_metrics = self.validate(epoch, self.stylized_imagenet_val_loader)
                sketch_imagenet_val_metrics = self.validate(epoch, self.sketch_imagenet_val_loader)
                imagenet_A_val_metrics = self.validate(epoch, self.imagenet_A_val_loader)
                imagenet_R_val_metrics = self.validate(epoch, self.imagenet_R_val_loader)
                # ImageNet-v2 variants
                imagenet_v2_matched_freq_metrics = self.validate(epoch, self.imagenet_v2_matched_freq_loader)
                imagenet_v2_threshold_metrics = self.validate(epoch, self.imagenet_v2_threshold_loader)
                imagenet_v2_top_images_metrics = self.validate(epoch, self.imagenet_v2_top_images_loader)
                # Imagenet C
                # imagenet_C_corruption_results = self.validate_imagenet_c_by_corruption(epoch)
                # imagenet_C_metrics = self.calculate_imagenet_c_metrics(imagenet_C_corruption_results)

                self.logger.info(
                    f"Val - Loss: {val_metrics['loss']:.4f}, "
                    f"Acc@1: {val_metrics['acc1']:.2f}, "
                    f"Acc@5: {val_metrics['acc5']:.2f}, "
                    f"Stylized IN Val - Loss: {stylized_imagenet_val_metrics['loss']:.4f}, "
                    f"Stylized IN Acc@1: {stylized_imagenet_val_metrics['acc1']:.2f}, "
                    f"Stylized IN Acc@5: {stylized_imagenet_val_metrics['acc5']:.2f}, "
                    f"Sketch IN Val - Loss: {sketch_imagenet_val_metrics['loss']:.4f}, "
                    f"Sketch IN Acc@1: {sketch_imagenet_val_metrics['acc1']:.2f}, "
                    f"Sketch IN Acc@5: {sketch_imagenet_val_metrics['acc5']:.2f}, "
                    f"Imagenet_A Val - Loss: {imagenet_A_val_metrics['loss']:.4f}, "
                    f"Imagenet_A Acc@1: {imagenet_A_val_metrics['acc1']:.2f}, "
                    f"Imagenet_A Acc@5: {imagenet_A_val_metrics['acc5']:.2f}, "
                    f"Imagenet_R Val - Loss: {imagenet_R_val_metrics['loss']:.4f}, "
                    f"Imagenet_R Acc@1: {imagenet_R_val_metrics['acc1']:.2f}, "
                    f"Imagenet_R Acc@5: {imagenet_R_val_metrics['acc5']:.2f},"
                    f"Imagenet_v2 matched_freq Val - Loss: {imagenet_v2_matched_freq_metrics['loss']:.4f}, "
                    f"Imagenet_v2 matched_freq Acc@1: {imagenet_v2_matched_freq_metrics['acc1']:.2f}, "
                    f"Imagenet_v2 matched_freq Acc@5: {imagenet_v2_matched_freq_metrics['acc5']:.2f},"
                    f"Imagenet_v2 threshold_metrics Val - Loss: {imagenet_v2_threshold_metrics['loss']:.4f}, "
                    f"Imagenet_v2 threshold_metrics Acc@1: {imagenet_v2_threshold_metrics['acc1']:.2f}, "
                    f"Imagenet_v2 threshold_metrics Acc@5: {imagenet_v2_threshold_metrics['acc5']:.2f},"
                    f"Imagenet_v2 top_images Val - Loss: {imagenet_v2_top_images_metrics['loss']:.4f}, "
                    f"Imagenet_v2 top_images Acc@1: {imagenet_v2_top_images_metrics['acc1']:.2f}, "
                    f"Imagenet_v2 top_images Acc@5: {imagenet_v2_top_images_metrics['acc5']:.2f},"
                )
                # # Imagenet C metrics
                # for metric, metric_value in imagenet_C_metrics.items():
                #     if isinstance(metric_value, dict):
                #         for k, v in metric_value.items():
                #             self.logger.info(
                #                 f"Imagenet_C corruption: {k}: {v:.4f}"
                #             )
                #     else:
                #         self.logger.info(
                #             f"Imagenet_C {metric}: {metric_value:.4f}"
                #         )

                # Log to wandb
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'imagenet_val/loss': val_metrics['loss'],
                        'imagenet_val/acc1': val_metrics['acc1'],
                        'imagenet_val/acc5': val_metrics['acc5'],
                        'stylized_IN_val/loss': stylized_imagenet_val_metrics['loss'],
                        'stylized_IN_val/acc1': stylized_imagenet_val_metrics['acc1'],
                        'stylized_IN_val/acc5': stylized_imagenet_val_metrics['acc5'],
                        'sketch_IN_val/loss': sketch_imagenet_val_metrics['loss'],
                        'sketch_IN_val/acc1': sketch_imagenet_val_metrics['acc1'],
                        'sketch_IN_val/acc5': sketch_imagenet_val_metrics['acc5'],
                        'imagenet_A/loss': imagenet_A_val_metrics['loss'],
                        'imagenet_A/acc1': imagenet_A_val_metrics['acc1'],
                        'imagenet_A/acc5': imagenet_A_val_metrics['acc5'],
                        'imagenet_R/loss': imagenet_R_val_metrics['loss'],
                        'imagenet_R/acc1': imagenet_R_val_metrics['acc1'],
                        'imagenet_R/acc5': imagenet_R_val_metrics['acc5'],
                        'imagenet_v2_matched_freq/loss': imagenet_v2_matched_freq_metrics['loss'],
                        'imagenet_v2_matched_freq/acc1': imagenet_v2_matched_freq_metrics['acc1'],
                        'imagenet_v2_matched_freq/acc5': imagenet_v2_matched_freq_metrics['acc5'],
                        'imagenet_v2_threshold/loss': imagenet_v2_threshold_metrics['loss'],
                        'imagenet_v2_threshold/acc1': imagenet_v2_threshold_metrics['acc1'],
                        'imagenet_v2_threshold/acc5': imagenet_v2_threshold_metrics['acc5'],
                        'imagenet_v2_top_images/loss': imagenet_v2_top_images_metrics['loss'],
                        'imagenet_v2_top_images/acc1': imagenet_v2_top_images_metrics['acc1'],
                        'imagenet_v2_top_images/acc5': imagenet_v2_top_images_metrics['acc5'],
                        'epoch': epoch
                    })
                    # # Imagenet C metrics
                    # imagenet_C_metrics_to_log = {}
                    # for metric, metric_value in imagenet_C_metrics.items():
                    #     if isinstance(metric_value, dict):
                    #         for k, v in metric_value.items():
                    #             imagenet_C_metrics_to_log['imagenet_C/' + k] = v
                    #     else:
                    #         imagenet_C_metrics_to_log['imagenet_C/' + metric] = metric_value
                    # wandb.log(imagenet_C_metrics_to_log)

            # Training
            train_metrics = self.train_epoch(epoch, weight_tracker)

            # Validation
            val_metrics = self.validate(epoch, self.val_loader)
            stylized_imagenet_val_metrics = self.validate(epoch, self.stylized_imagenet_val_loader)
            sketch_imagenet_val_metrics = self.validate(epoch, self.sketch_imagenet_val_loader)
            imagenet_A_val_metrics = self.validate(epoch, self.imagenet_A_val_loader)
            imagenet_R_val_metrics = self.validate(epoch, self.imagenet_R_val_loader)
            # ImageNet-v2 variants
            imagenet_v2_matched_freq_metrics = self.validate(epoch, self.imagenet_v2_matched_freq_loader)
            imagenet_v2_threshold_metrics = self.validate(epoch, self.imagenet_v2_threshold_loader)
            imagenet_v2_top_images_metrics = self.validate(epoch, self.imagenet_v2_top_images_loader)
            # Imagenet C
            # imagenet_C_corruption_results = self.validate_imagenet_c_by_corruption(epoch)
            # imagenet_C_metrics = self.calculate_imagenet_c_metrics(imagenet_C_corruption_results)

            # Update scheduler
            if epoch <= self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()

            # Log epoch results
            self.logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc@1: {train_metrics['acc1']:.2f}, "
                f"Acc@5: {train_metrics['acc5']:.2f}"
            )
            self.logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Acc@1: {val_metrics['acc1']:.2f}, "
                f"Acc@5: {val_metrics['acc5']:.2f}, "
                f"Stylized IN Val - Loss: {stylized_imagenet_val_metrics['loss']:.4f}, "
                f"Stylized IN Acc@1: {stylized_imagenet_val_metrics['acc1']:.2f}, "
                f"Stylized IN Acc@5: {stylized_imagenet_val_metrics['acc5']:.2f}, "
                f"Sketch IN Val - Loss: {sketch_imagenet_val_metrics['loss']:.4f}, "
                f"Sketch IN Acc@1: {sketch_imagenet_val_metrics['acc1']:.2f}, "
                f"Sketch IN Acc@5: {sketch_imagenet_val_metrics['acc5']:.2f}, "
                f"Imagenet_A Val - Loss: {imagenet_A_val_metrics['loss']:.4f}, "
                f"Imagenet_A Acc@1: {imagenet_A_val_metrics['acc1']:.2f}, "
                f"Imagenet_A Acc@5: {imagenet_A_val_metrics['acc5']:.2f}, "
                f"Imagenet_R Val - Loss: {imagenet_R_val_metrics['loss']:.4f}, "
                f"Imagenet_R Acc@1: {imagenet_R_val_metrics['acc1']:.2f}, "
                f"Imagenet_R Acc@5: {imagenet_R_val_metrics['acc5']:.2f},"
                f"Imagenet_v2 matched_freq Val - Loss: {imagenet_v2_matched_freq_metrics['loss']:.4f}, "
                f"Imagenet_v2 matched_freq Acc@1: {imagenet_v2_matched_freq_metrics['acc1']:.2f}, "
                f"Imagenet_v2 matched_freq Acc@5: {imagenet_v2_matched_freq_metrics['acc5']:.2f},"
                f"Imagenet_v2 threshold_metrics Val - Loss: {imagenet_v2_threshold_metrics['loss']:.4f}, "
                f"Imagenet_v2 threshold_metrics Acc@1: {imagenet_v2_threshold_metrics['acc1']:.2f}, "
                f"Imagenet_v2 threshold_metrics Acc@5: {imagenet_v2_threshold_metrics['acc5']:.2f},"
                f"Imagenet_v2 top_images Val - Loss: {imagenet_v2_top_images_metrics['loss']:.4f}, "
                f"Imagenet_v2 top_images Acc@1: {imagenet_v2_top_images_metrics['acc1']:.2f}, "
                f"Imagenet_v2 top_images Acc@5: {imagenet_v2_top_images_metrics['acc5']:.2f},"
            )
            # # Imagenet C metrics
            # for metric, metric_value in imagenet_C_metrics.items():
            #     if isinstance(metric_value, dict):
            #         for k,v in metric_value.items():
            #             self.logger.info(
            #                 f"Imagenet_C corruption: {k}: {v:.4f}"
            #             )
            #     else:
            #         self.logger.info(
            #             f"Imagenet_C {metric}: {metric_value:.4f}"
            #         )


            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'imagenet_val/loss': val_metrics['loss'],
                    'imagenet_val/acc1': val_metrics['acc1'],
                    'imagenet_val/acc5': val_metrics['acc5'],
                    'stylized_IN_val/loss': stylized_imagenet_val_metrics['loss'],
                    'stylized_IN_val/acc1': stylized_imagenet_val_metrics['acc1'],
                    'stylized_IN_val/acc5': stylized_imagenet_val_metrics['acc5'],
                    'sketch_IN_val/loss': sketch_imagenet_val_metrics['loss'],
                    'sketch_IN_val/acc1': sketch_imagenet_val_metrics['acc1'],
                    'sketch_IN_val/acc5': sketch_imagenet_val_metrics['acc5'],
                    'imagenet_A/loss': imagenet_A_val_metrics['loss'],
                    'imagenet_A/acc1': imagenet_A_val_metrics['acc1'],
                    'imagenet_A/acc5': imagenet_A_val_metrics['acc5'],
                    'imagenet_R/loss': imagenet_R_val_metrics['loss'],
                    'imagenet_R/acc1': imagenet_R_val_metrics['acc1'],
                    'imagenet_R/acc5': imagenet_R_val_metrics['acc5'],
                    'imagenet_v2_matched_freq/loss': imagenet_v2_matched_freq_metrics['loss'],
                    'imagenet_v2_matched_freq/acc1': imagenet_v2_matched_freq_metrics['acc1'],
                    'imagenet_v2_matched_freq/acc5': imagenet_v2_matched_freq_metrics['acc5'],
                    'imagenet_v2_threshold/loss': imagenet_v2_threshold_metrics['loss'],
                    'imagenet_v2_threshold/acc1': imagenet_v2_threshold_metrics['acc1'],
                    'imagenet_v2_threshold/acc5': imagenet_v2_threshold_metrics['acc5'],
                    'imagenet_v2_top_images/loss': imagenet_v2_top_images_metrics['loss'],
                    'imagenet_v2_top_images/acc1': imagenet_v2_top_images_metrics['acc1'],
                    'imagenet_v2_top_images/acc5': imagenet_v2_top_images_metrics['acc5'],
                    'epoch': epoch
                })
                # # Imagenet C metrics
                # imagenet_C_metrics_to_log = {}
                # for metric, metric_value in imagenet_C_metrics.items():
                #     if isinstance(metric_value, dict):
                #         for k, v in metric_value.items():
                #             imagenet_C_metrics_to_log['imagenet_C/' + k] = v
                #     else:
                #         imagenet_C_metrics_to_log['imagenet_C/' + metric] = metric_value
                # wandb.log(imagenet_C_metrics_to_log)

            # Save checkpoint
            is_best = val_metrics['acc1'] > best_acc1
            if is_best:
                best_acc1 = val_metrics['acc1']

            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping (optional)
            if self.config.get('early_stopping', False):
                # Implement early stopping logic here
                pass

        self.logger.info(f"Training completed! Best Acc@1: {best_acc1:.2f}")

        if self.config.get('use_wandb', False):
            wandb.finish()


# Example training script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dense Contrastive ViT')
    parser.add_argument('--imagenet', type=str,
                        default='/home/cognition/datasets/IMAGENET/imagenet_ILSVRC-2012_ImageNet-1K',
                        help='Path to ImageNet validation set')
    parser.add_argument('--stylized-imagenet', type=str,
                        default='/home/cognition/datasets/stylized_imagenet',
                        help='Path to Stylized ImageNet validation set')
    parser.add_argument('--imagenet-A', type=str,
                        default='/home/cognition/datasets/imagenet-a',
                        help='Path to ImageNet_A validation set')
    parser.add_argument('--imagenet-R', type=str,
                        default='/home/cognition/datasets/imagenet-r',
                        help='Path to ImageNet_R validation set')
    parser.add_argument('--imagenet-v2', type=str,
                        default='/home/cognition/datasets/imagenet-v2',
                        help='Path to ImageNet_v2 validation set')
    parser.add_argument('--imagenet-C', type=str,
                        default='/home/cognition/datasets/imagenet-c',
                        help='Path to ImageNet_C validation set')
    parser.add_argument('--model-name', type=str, default='vit_tiny_patch16_224', help='Model architecture')
    parser.add_argument('--experiment-name', type=str, default='dense_vit_tiny_baseline1', help='WANDB experiment name')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=min(4 * torch.cuda.device_count(), os.cpu_count() // 2), help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--queue-size', type=int, default=1000, help='Length of neg queue size')
    parser.add_argument('--contrastive-weight-adaptive',  action="store_true", help='If to use adaptive contrastive weighting. If this is True, --lamdba-weight parameter is ignored.')
    parser.add_argument('--no-log-baseline-res', action="store_true", help='Logs results before training begins with the existing weights')
    parser.add_argument('--correspondence-features', type=str, default='dense', help='What features to use for correspondence finding. Options: [dense, backbone]')
    parser.add_argument('--lambda-weight', type=float, default=0.5, help='Lambda to weight class and dense loss. If 0, total loss = class loss, If 1, total loss = dense loss')
    parser.add_argument('--grad-clip', type=float, default=5, help='At what value to clip and scale the gradients')
    parser.add_argument('--learning-rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--ckpt-dir', type=str, default='./models', help='Directory to save results')
    parser.add_argument('--pretrained', type=bool, required=False, default=True, help='If to use deit pretrained weights')
    parser.add_argument('--model-parallel', type=bool, required=False, default=False, help='If to parallelize the model across GPUs')

    args = parser.parse_args()
    # Configuration
    config = {
        # Model arch arguments
        'model_name': args.model_name,
        'num_classes': args.num_classes,
        'dense_dim': 128,
        'model_parallel': args.model_parallel,

        # Dataset related arguments
        'data_path': args.imagenet,  # Update this path
        'stylized_imagenet': args.stylized_imagenet,
        'imagenet_A': args.imagenet_A,
        'imagenet_R': args.imagenet_R,
        'imagenet_C': args.imagenet_C,
        'imagenet_v2': args.imagenet_v2,
        'save_dir': args.ckpt_dir,

        # Optimization arguments
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.05,
        'warmup_epochs': 5,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'num_workers': args.num_workers,
        'grad_clip': args.grad_clip,
        'pretrained': args.pretrained,
        'temperature': 0.2,
        'queue_size': args.queue_size,
        'momentum': 0.999,

        # Contrastive learning arguments
        'contrastive_weight_adaptive': args.contrastive_weight_adaptive,
        'correspondence_features': args.correspondence_features,
        'lambda_weight': args.lambda_weight,

        # Logging related arguments
        'use_wandb': True,
        'wandb_project': 'dense-contrastive-vit',
        'experiment_name': args.experiment_name,
        'log_interval': 100,
        'no_log_baseline_res': args.no_log_baseline_res,

    }
    print(f"Args:{args}")

    # Create trainer and start training
    trainer = DenseContrastiveTrainer(config)
    trainer.train()