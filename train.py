import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import time
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, Optional

from dense_vit_model import DenseContrastiveViT, DenseContrastiveLoss


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
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'dense-contrastive-vit'),
                config=config,
                name=config.get('experiment_name', 'dense_vit_tiny_baseline')
            )

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
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")

    def setup_data(self):
        """Setup data loaders"""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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

        # Create datasets
        train_dataset = ImageFolder(
            root=os.path.join(self.config['data_path'], 'train'),
            transform=train_transform
        )

        val_dataset = ImageFolder(
            root=os.path.join(self.config['data_path'], 'val'),
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

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")

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
            momentum=self.config.get('momentum', 0.999)
        ).to(self.device)

        # Loss weight
        self.lambda_weight = self.config.get('lambda_weight', 0.5)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
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

        end = time.time()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for i, (images, targets) in enumerate(pbar):
            # Measure data loading time
            data_time.update(time.time() - end)

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Create two augmented views for contrastive learning
            # For now, we'll use the same image twice - in practice, you'd want different augmentations
            images_1 = images
            images_2 = images  # TODO: Apply different augmentations

            # Forward pass
            cls_output_1, dense_features_1, backbone_features_1 = self.model(images_1, return_dense=True)
            cls_output_2, dense_features_2, backbone_features_2 = self.model(images_2, return_dense=True)
            # cls_output_1.shape: [64, 1000], dense_features_1.shape: [64, 14, 14, 128]
            # cls_output_2.shape: [64, 1000], dense_features_2.shape: [64, 14, 14, 128]

            # Classification loss (using both views)
            cls_loss = (self.classification_criterion(cls_output_1, targets) +
                        self.classification_criterion(cls_output_2, targets)) / 2

            # Dense contrastive loss
            # For correspondence, we use the dense features as backbone features
            # In practice, you might want to use features from an earlier layer
            dense_loss = self.dense_contrastive_criterion(
                dense_features_1, dense_features_2,
                backbone_features_1, backbone_features_2  # Using same features for correspondence
            )

            # Total loss
            total_loss = (1 - self.lambda_weight) * cls_loss + self.lambda_weight * dense_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.optimizer.step()

            # Measure accuracy
            acc1, acc5 = self.accuracy(cls_output_1, targets, topk=(1, 5))

            # Update metrics
            losses.update(total_loss.item(), images.size(0))
            cls_losses.update(cls_loss.item(), images.size(0))
            dense_losses.update(dense_loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

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
                    'epoch': epoch
                })

        return {
            'loss': losses.avg,
            'cls_loss': cls_losses.avg,
            'dense_loss': dense_losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')

            for i, (images, targets) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass (no dense features for validation)
                cls_output = self.model(images, return_dense=False)

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
        checkpoint_path = Path(self.config['save_dir']) / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = Path(self.config['save_dir']) / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best checkpoint saved at epoch {epoch}")

    def train(self):
        """Main training loop"""
        best_acc1 = 0.0

        for epoch in range(1, self.config['epochs'] + 1):
            self.logger.info(f"Epoch {epoch}/{self.config['epochs']}")

            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate(epoch)

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
                f"Acc@5: {val_metrics['acc5']:.2f}"
            )

            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'val/loss': val_metrics['loss'],
                    'val/acc1': val_metrics['acc1'],
                    'val/acc5': val_metrics['acc5'],
                    'epoch': epoch
                })

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
    parser.add_argument('--imagenet-train', type=str,
                        default='/home/cognition/datasets/IMAGENET/imagenet_ILSVRC-2012_ImageNet-1K',
                        help='Path to ImageNet validation set')
    parser.add_argument('--model-name', type=str, default='vit_tiny_patch16_224', help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--ckpt-dir', type=str, default='./models', help='Directory to save results')
    parser.add_argument('--pretrained', type=bool, required=False, default=True,
                        help='If to use deit pretrained weights')

    args = parser.parse_args()
    # Configuration
    config = {
        'model_name': args.model_name,
        'num_classes': args.num_classes,
        'dense_dim': 128,
        'data_path': args.imagenet_train,  # Update this path
        'save_dir': args.ckpt_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': 1e-3,
        'weight_decay': 0.05,
        'warmup_epochs': 5,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'num_workers': args.num_workers,
        'grad_clip': 1.0,
        'lambda_weight': 0.5,
        'temperature': 0.2,
        'queue_size': 1000,
        'momentum': 0.999,
        'use_wandb': True,
        'wandb_project': 'dense-contrastive-vit',
        'experiment_name': 'dense_vit_tiny_baseline',
        'log_interval': 100
    }

    # Create trainer and start training
    trainer = DenseContrastiveTrainer(config)
    trainer.train()