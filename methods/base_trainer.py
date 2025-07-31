"""
Base trainer class providing common functionality for OOD detection methods.

This class contains shared functionality like data loading, model setup, 
evaluation, and common training utilities.
"""

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import sys
from abc import ABC, abstractmethod

from models.wrn import WideResNet
from utils.display_results import get_measures, print_measures


class BaseTrainer(ABC):
    """Base trainer class for OOD detection methods."""
    
    def __init__(self, args):
        """Initialize base trainer with arguments."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup basic training components
        self.setup_data_loaders()
        self.setup_model()
        self.setup_optimizer()
        
        # Evaluation utilities
        self.concat = lambda x: np.concatenate(x, axis=0)
        self.to_np = lambda x: x.data.cpu().numpy()
        
        print(f"Initialized {self.__class__.__name__} for {args.dataset.upper()}")
    
    def setup_data_loaders(self):
        """Setup data loaders for training and testing."""
        try:
            from data.cifar_loaders import get_cifar_loaders
            from data.auxiliary_loaders import get_auxiliary_loader  
            from data.test_loaders import get_test_loaders
            
            # Get ID data loaders
            self.train_loader_in, self.test_loader, self.num_classes = get_cifar_loaders(
                dataset=self.args.dataset,
                batch_size=self.args.batch_size,
                test_batch_size=self.args.test_bs,
                num_workers=self.args.num_workers
            )
            
            # Get auxiliary OOD loader (method-specific)
            self.train_loader_out = get_auxiliary_loader(
                method=self.args.method,
                batch_size=self.args.oe_batch_size,
                num_workers=self.args.num_workers
            )
            
            # Get test OOD loaders
            self.test_loaders = get_test_loaders(
                datasets=self.args.test_datasets,
                batch_size=self.args.test_bs,
                num_workers=self.args.num_workers
            )
            
        except ImportError as e:
            print(f"Error importing data loaders: {e}")
            print("Using fallback data loading...")
            self._setup_fallback_data_loaders()
    
    def _setup_fallback_data_loaders(self):
        """Fallback data loading using existing code."""
        import torchvision.transforms as trn
        import torchvision.datasets as dset
        
        # Basic CIFAR transforms
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        
        train_transform = trn.Compose([
            trn.RandomHorizontalFlip(), 
            trn.RandomCrop(32, padding=4),
            trn.ToTensor(), 
            trn.Normalize(mean, std)
        ])
        test_transform = trn.Compose([
            trn.ToTensor(), 
            trn.Normalize(mean, std)
        ])
        
        # CIFAR data
        if self.args.dataset == 'cifar10':
            train_data_in = dset.CIFAR10('./data/cifarpy', train=True, 
                                       transform=train_transform, download=True)
            test_data = dset.CIFAR10('./data/cifarpy', train=False, 
                                   transform=test_transform, download=True)
            self.num_classes = 10
        else:
            train_data_in = dset.CIFAR100('./data/cifarpy', train=True, 
                                        transform=train_transform, download=True)
            test_data = dset.CIFAR100('./data/cifarpy', train=False, 
                                    transform=test_transform, download=True)
            self.num_classes = 100
        
        self.train_loader_in = torch.utils.data.DataLoader(
            train_data_in, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self.args.test_bs, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True
        )
        
        # Auxiliary OOD data (method-specific)
        if self.args.method == 'wdoe':
            # TinyImageNet-200 for W-DOE
            ood_data = dset.ImageFolder(
                root="./data/tiny-imagenet-200/train/",
                transform=trn.Compose([
                    trn.Resize(32), trn.RandomCrop(32, padding=4), 
                    trn.RandomHorizontalFlip(), trn.ToTensor(), 
                    trn.Normalize(mean, std)
                ])
            )
        else:
            # 80M Tiny Images for DAL
            try:
                from utils.tinyimages_80mn_loader import TinyImages
                ood_data = TinyImages(
                    transform=trn.Compose([
                        trn.ToTensor(), trn.ToPILImage(), 
                        trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), 
                        trn.ToTensor(), trn.Normalize(mean, std)
                    ]), 
                    exclude_cifar=True
                )
            except:
                print("Warning: Could not load 80M Tiny Images, using TinyImageNet as fallback")
                ood_data = dset.ImageFolder(
                    root="./data/tiny-imagenet-200/train/",
                    transform=trn.Compose([
                        trn.Resize(32), trn.RandomCrop(32, padding=4), 
                        trn.RandomHorizontalFlip(), trn.ToTensor(), 
                        trn.Normalize(mean, std)
                    ])
                )
        
        self.train_loader_out = torch.utils.data.DataLoader(
            ood_data, batch_size=self.args.oe_batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True
        )
        
        # Test OOD loaders (basic setup)
        self.test_loaders = {}
        # This will be setup by specific trainer classes
    
    def setup_model(self):
        """Setup the WideResNet model."""
        self.model = WideResNet(
            depth=self.args.layers,
            num_classes=self.num_classes,
            widen_factor=self.args.widen_factor,
            dropRate=self.args.droprate
        ).to(self.device)
        
        # Load pretrained weights if available
        self.load_pretrained_model()
        
        print(f"Model: WideResNet-{self.args.layers}-{self.args.widen_factor}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_pretrained_model(self):
        """Load pretrained model weights."""
        if self.args.dataset == 'cifar10':
            model_path = './ckpt/cifar10_wrn_pretrained_epoch_99.pt'
        else:
            model_path = './ckpt/cifar100_wrn_pretrained_epoch_99.pt'
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Training from scratch...")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True
        )
        
        # Cosine annealing scheduler
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
        
        total_steps = self.args.epochs * len(self.train_loader_in)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lambda step: cosine_annealing(
                step, total_steps, 1, 1e-6 / self.args.learning_rate
            )
        )
    
    def get_ood_scores(self, loader, in_dist=False):
        """Get OOD scores for evaluation."""
        scores = []
        self.model.eval()
        
        ood_num_examples = len(self.test_loader.dataset) // 5
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= ood_num_examples // self.args.test_bs and not in_dist:
                    break
                
                data = data.to(self.device)
                output = self.model(data)
                
                # Use unified evaluation mode
                if self.args.eval_mode == 'softmax':
                    smax = self.to_np(F.softmax(output, dim=1))
                else:  # logits mode
                    smax = self.to_np(output)
                
                scores.append(-np.max(smax, axis=1))
        
        if in_dist:
            return self.concat(scores).copy()
        else:
            return self.concat(scores)[:ood_num_examples].copy()
    
    def evaluate_ood(self, ood_loader, in_score):
        """Evaluate OOD detection performance."""
        aurocs, auprs, fprs = [], [], []
        
        for _ in range(self.args.num_runs):
            out_score = self.get_ood_scores(ood_loader)
            
            if self.args.out_as_pos:
                measures = get_measures(out_score, in_score)
            else:
                measures = get_measures(-in_score, -out_score)
            
            aurocs.append(measures[0])
            auprs.append(measures[1]) 
            fprs.append(measures[2])
        
        auroc = np.mean(aurocs)
        aupr = np.mean(auprs)
        fpr = np.mean(fprs)
        
        return fpr, auroc, aupr
    
    def test(self):
        """Test the model on all OOD datasets."""
        print("Starting evaluation...")
        self.model.eval()
        
        # Get ID scores
        in_score = self.get_ood_scores(self.test_loader, in_dist=True)
        
        results = {}
        print(f"\nOOD Detection Results ({self.args.eval_mode} scoring):")
        print("-" * 60)
        
        # Test on each OOD dataset
        for dataset_name in self.args.test_datasets:
            if hasattr(self, 'test_loaders') and dataset_name in self.test_loaders:
                loader = self.test_loaders[dataset_name]
                fpr, auroc, aupr = self.evaluate_ood(loader, in_score)
                results[dataset_name] = (fpr, auroc, aupr)
                
                print(f"{dataset_name.upper():<12}: FPR95={fpr:.2f}, AUROC={auroc:.2f}, AUPR={aupr:.2f}")
            elif dataset_name == 'cifar_cross':
                # CIFAR cross-evaluation
                try:
                    self._test_cifar_cross(in_score, results)
                except Exception as e:
                    print(f"Error in CIFAR cross-evaluation: {e}")
        
        return results
    
    def _test_cifar_cross(self, in_score, results):
        """Test CIFAR cross-evaluation (CIFAR-10 vs CIFAR-100)."""
        import torchvision.datasets as dset
        import torchvision.transforms as trn
        
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        
        if self.args.dataset == 'cifar10':
            cross_data = dset.CIFAR100('../data/cifarpy', train=False, 
                                     transform=test_transform, download=True)
        else:
            cross_data = dset.CIFAR10('../data/cifarpy', train=False, 
                                    transform=test_transform, download=True)
        
        cross_loader = torch.utils.data.DataLoader(
            cross_data, batch_size=self.args.test_bs, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True
        )
        
        fpr, auroc, aupr = self.evaluate_ood(cross_loader, in_score)
        results['cifar_cross'] = (fpr, auroc, aupr)
        print(f"{'CIFAR_CROSS':<12}: FPR95={fpr:.2f}, AUROC={auroc:.2f}, AUPR={aupr:.2f}")
    
    @abstractmethod
    def train_epoch(self, epoch):
        """Train for one epoch (method-specific implementation)."""
        pass
    
    def train_and_test(self):
        """Complete training and testing pipeline."""
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            
            if self.args.verbose or epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.args.epochs}, Loss: {epoch_loss:.4f}")
            
            # Test accuracy on ID data
            if epoch % 20 == 0 or epoch == self.args.epochs:
                acc = self.test_accuracy()
                print(f"Epoch {epoch}: ID Accuracy = {acc:.2f}%")
        
        print("Training completed. Starting evaluation...")
        return self.test()
    
    def test_accuracy(self):
        """Test accuracy on ID data."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total