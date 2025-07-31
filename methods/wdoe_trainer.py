"""
W-DOE (Wasserstein Distribution-agnostic Outlier Exposure) Trainer

This module implements the W-DOE training procedure using Adversarial Weight Perturbation (AWP)
for implicit data synthesis to improve OOD detection performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

from .base_trainer import BaseTrainer
from models.wrn import WideResNet
import utils.utils_awp as awp


class WDOETrainer(BaseTrainer):
    """W-DOE trainer implementing implicit data synthesis via AWP."""
    
    def __init__(self, args):
        """Initialize W-DOE trainer."""
        super().__init__(args)
        
        # W-DOE specific setup
        self.gamma = args.gamma if hasattr(args, 'gamma') else 0.5
        self.warmup = args.warmup if hasattr(args, 'warmup') else 5
        self.begin_epoch = getattr(args, 'begin_epoch', 0)
        
        # Initialize weight difference tracking
        self.diff = None
        
        print(f"W-DOE Config: gamma={self.gamma}, warmup={self.warmup}")
    
    def train_epoch(self, epoch):
        """Train one epoch using W-DOE methodology."""
        # Create proxy network for AWP
        proxy = WideResNet(
            depth=self.args.layers,
            num_classes=self.num_classes, 
            widen_factor=self.args.widen_factor,
            dropRate=0.0  # No dropout for proxy
        ).to(self.device)
        proxy_optim = torch.optim.SGD(proxy.parameters(), lr=1.0)
        
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Synchronize auxiliary data offset
        if hasattr(self.train_loader_out.dataset, 'offset'):
            self.train_loader_out.dataset.offset = np.random.randint(len(self.train_loader_in.dataset))
        
        for batch_idx, (in_set, out_set) in enumerate(zip(self.train_loader_in, self.train_loader_out)):
            # Combine ID and OOD data
            data = torch.cat((in_set[0], out_set[0]), 0).to(self.device)
            target = in_set[1].to(self.device)
            
            # W-DOE implicit data synthesis (after warmup)
            if epoch >= self.warmup and self.diff is not None:
                # Load proxy with current model weights
                proxy.load_state_dict(self.model.state_dict())
                proxy.train()
                
                # Get original embeddings for regularization
                org_embs = self.model.feature_list(data)[1].detach()[len(in_set[0]):]
                
                # Apply weight perturbation to proxy
                try:
                    awp.add_into_weights(proxy, self.diff, coeff=self.gamma)
                except:
                    pass
                
                # Dynamic gamma sampling
                gamma_sample = torch.tensor([1e-1, 1e-2, 1e-3, 1e-4])[torch.randperm(4)][0]
                scale = torch.tensor([1.0], device=self.device, requires_grad=True)
                
                # Forward pass through proxy
                logits, embs = proxy.feature_list(data)
                x = logits * scale
                
                # Surrogate loss and regularization
                l_sur = (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
                l_reg = ((org_embs - embs[-1]) ** 2).mean(-1).mean()
                reg_sur = torch.sum(torch.autograd.grad(l_sur, [scale], create_graph=True)[0] ** 2)
                
                # Update proxy
                proxy_optim.zero_grad()
                (reg_sur + self.gamma * l_reg).backward()
                torch.nn.utils.clip_grad_norm_(proxy.parameters(), 1.0)
                proxy_optim.step()
                
                # Update weight difference
                if epoch == self.warmup and batch_idx == 0:
                    self.diff = awp.diff_in_weights(self.model, proxy)
                else:
                    self.diff = awp.average_diff(self.diff, awp.diff_in_weights(self.model, proxy), beta=0.6)
                
                # Apply perturbation to main model
                awp.add_into_weights(self.model, self.diff, coeff=gamma_sample)
            
            # Main training step
            x = self.model(data)
            l_ce = F.cross_entropy(x[:len(in_set[0])], target)
            l_oe = -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
            
            # Loss scheduling based on epoch and dataset
            if epoch >= self.warmup:
                loss = l_oe  # Focus on OE loss after warmup
            else:
                loss = l_ce + l_oe  # Combined loss during warmup
            
            # Main model update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Additional classification loss (if after warmup)
            if epoch >= self.warmup:
                # Additional ID classification step
                self.optimizer.zero_grad()
                x = self.model(data)
                l_ce = F.cross_entropy(x[:len(in_set[0])], target)
                l_ce.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Progress reporting
            if batch_idx % 100 == 0 and self.args.verbose:
                sys.stdout.write(f'\rEpoch {epoch} [{batch_idx}/{len(self.train_loader_in)}] '
                               f'Loss: {loss.item():.4f} (CE: {l_ce.item():.4f}, OE: {l_oe.item():.4f})')
        
        if self.args.verbose:
            print()  # New line after progress
        
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def setup_test_loaders(self):
        """Setup test loaders with W-DOE specific datasets."""
        try:
            from data.test_loaders import get_test_loaders
            
            # W-DOE uses comprehensive test datasets including Places365 and LSUN
            test_datasets = ['dtd', 'svhn', 'isun', 'cifar_cross', 'places365', 'lsun_c', 'lsun_r']
            
            self.test_loaders = get_test_loaders(
                datasets=test_datasets,
                batch_size=self.args.test_bs,
                num_workers=self.args.num_workers
            )
                
        except ImportError:
            # Fallback to manual setup
            self._setup_fallback_test_loaders()
    
    def _setup_fallback_test_loaders(self):
        """Fallback test loader setup using existing code."""
        import torchvision.transforms as trn
        import torchvision.datasets as dset
        import utils.svhn_loader as svhn
        
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        
        self.test_loaders = {}
        
        # Textures (DTD)
        try:
            texture_data = dset.ImageFolder(
                root="./data/dtd/images",
                transform=trn.Compose([
                    trn.Resize(32), trn.CenterCrop(32), 
                    trn.ToTensor(), trn.Normalize(mean, std)
                ])
            )
            self.test_loaders['dtd'] = torch.utils.data.DataLoader(
                texture_data, batch_size=self.args.test_bs, shuffle=True,
                num_workers=4, pin_memory=False
            )
        except:
            pass
        
        # SVHN
        try:
            svhn_data = svhn.SVHN(
                root='./data/svhn/', split="test",
                transform=test_transform, download=True
            )
            self.test_loaders['svhn'] = torch.utils.data.DataLoader(
                svhn_data, batch_size=self.args.test_bs, shuffle=True,
                num_workers=4, pin_memory=False
            )
        except:
            pass
        
        # iSUN  
        try:
            isun_data = dset.ImageFolder(
                root="./data/iSUN", transform=test_transform
            )
            self.test_loaders['isun'] = torch.utils.data.DataLoader(
                isun_data, batch_size=self.args.test_bs, shuffle=True,
                num_workers=4, pin_memory=False
            )
        except:
            pass
        
        # Places365
        try:
            places365_data = dset.ImageFolder(
                root="./data/places365_standard/",
                transform=trn.Compose([
                    trn.Resize(32), trn.CenterCrop(32),
                    trn.ToTensor(), trn.Normalize(mean, std)
                ])
            )
            self.test_loaders['places365'] = torch.utils.data.DataLoader(
                places365_data, batch_size=self.args.test_bs, shuffle=True,
                num_workers=4, pin_memory=False
            )
        except:
            pass
        
        # LSUN-C
        try:
            lsunc_data = dset.ImageFolder(
                root="./data/LSUN",
                transform=trn.Compose([
                    trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)
                ])
            )
            self.test_loaders['lsun_c'] = torch.utils.data.DataLoader(
                lsunc_data, batch_size=self.args.test_bs, shuffle=True,
                num_workers=4, pin_memory=False
            )
        except:
            pass
        
        # LSUN-R
        try:
            lsunr_data = dset.ImageFolder(
                root="./data/LSUN_resize",
                transform=trn.Compose([
                    trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)
                ])
            )
            self.test_loaders['lsun_r'] = torch.utils.data.DataLoader(
                lsunr_data, batch_size=self.args.test_bs, shuffle=True,
                num_workers=4, pin_memory=False
            )
        except:
            pass
    
    def train_and_test(self):
        """Complete W-DOE training and testing pipeline."""
        print(f"Starting W-DOE training for {self.args.epochs} epochs...")
        print(f"Warmup period: {self.warmup} epochs")
        
        # Setup test loaders
        if not hasattr(self, 'test_loaders') or not self.test_loaders:
            self.setup_test_loaders()
        
        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            
            # Progress reporting
            if epoch <= self.warmup:
                stage = f"Warmup {epoch}/{self.warmup}"
            else:
                stage = f"Main {epoch - self.warmup}/{self.args.epochs - self.warmup}"
            
            if self.args.verbose or epoch % 5 == 0:
                print(f"Epoch {epoch} ({stage}): Loss = {epoch_loss:.4f}")
            
            # Periodic accuracy check
            if epoch % 10 == 0 or epoch == self.args.epochs:
                acc = self.test_accuracy()
                print(f"Epoch {epoch}: ID Accuracy = {acc:.2f}%")
        
        print("W-DOE training completed. Starting evaluation...")
        return self.test()
    
    def get_method_info(self):
        """Get method-specific information."""
        return {
            'method': 'W-DOE',
            'gamma': self.gamma,
            'warmup': self.warmup,
            'auxiliary_data': 'TinyImageNet-200',
            'key_technique': 'Adversarial Weight Perturbation (AWP)',
            'implicit_data_synthesis': True
        }