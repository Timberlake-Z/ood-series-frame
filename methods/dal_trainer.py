"""
DAL (Distributional-Augmented OOD Learning) Trainer

This module implements the DAL training procedure using embedding space augmentation
within a Wasserstein ball for improved OOD detection performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

from .base_trainer import BaseTrainer


class DALTrainer(BaseTrainer):
    """DAL trainer implementing distributional augmentation in embedding space."""
    
    def __init__(self, args):
        """Initialize DAL trainer."""
        super().__init__(args)
        
        # DAL specific parameters
        self.gamma = getattr(args, 'gamma', 10.0)
        self.beta = getattr(args, 'beta', 0.01 if args.dataset == 'cifar10' else 0.005)
        self.rho = getattr(args, 'rho', 10.0)
        self.strength = getattr(args, 'strength', 1.0)
        self.iter = getattr(args, 'iter', 10)
        self.warmup = getattr(args, 'warmup', 0)
        
        # Initialize adaptive gamma
        self.current_gamma = torch.tensor(self.gamma, device=self.device)
        
        print(f"DAL Config: gamma={self.gamma}, beta={self.beta}, rho={self.rho}, "
              f"strength={self.strength}, iter={self.iter}")
    
    def train_epoch(self, epoch):
        """Train one epoch using DAL methodology."""
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
            
            # Forward pass to get embeddings
            x, emb = self.model.pred_emb(data)
            
            # Standard losses
            l_ce = F.cross_entropy(x[:len(in_set[0])], target)
            l_oe_old = -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
            
            # DAL embedding augmentation
            emb_oe = emb[len(in_set[0]):].detach()
            emb_bias = torch.rand_like(emb_oe) * 0.0001
            
            # Iterative adversarial perturbation in embedding space
            for _ in range(self.iter):
                emb_bias.requires_grad_(True)
                
                # Augmented OOD predictions
                x_aug = self.model.fc(emb_bias + emb_oe)
                l_sur = -(x_aug.mean(1) - torch.logsumexp(x_aug, dim=1)).mean()
                r_sur = (emb_bias.abs()).mean(-1).mean()
                l_sur = l_sur - r_sur * self.current_gamma
                
                # Compute gradients
                grads = torch.autograd.grad(l_sur, [emb_bias])[0]
                grads = grads / (grads ** 2).sum(-1).sqrt().unsqueeze(1)
                
                # Update bias
                emb_bias = emb_bias.detach() + self.strength * grads.detach()
                self.optimizer.zero_grad()
            
            # Adaptive gamma update
            self.current_gamma -= self.beta * (self.rho - r_sur.detach())
            self.current_gamma = self.current_gamma.clamp(min=0.0, max=self.gamma)
            
            # Final augmented OOD loss
            if epoch >= self.warmup:
                x_oe = self.model.fc(emb[len(in_set[0]):] + emb_bias)
            else:
                x_oe = self.model.fc(emb[len(in_set[0]):])
            
            l_oe = -(x_oe.mean(1) - torch.logsumexp(x_oe, dim=1)).mean()
            
            # Combined loss
            loss = l_ce + 0.5 * l_oe
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Progress reporting
            if batch_idx % 100 == 0 and self.args.verbose:
                sys.stdout.write(f'\rEpoch {epoch} [{batch_idx}/{len(self.train_loader_in)}] '
                               f'Loss: {loss.item():.4f} (CE: {l_ce.item():.4f}, OE: {l_oe.item():.4f}, '
                               f'γ: {self.current_gamma.item():.3f})')
        
        if self.args.verbose:
            print()  # New line after progress
        
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def setup_test_loaders(self):
        """Setup test loaders with comprehensive DAL test datasets."""
        try:
            from data.test_loaders import get_test_loaders
            
            # DAL uses comprehensive test datasets including all standard benchmarks
            test_datasets = ['dtd', 'svhn', 'places365', 'lsun_c', 'lsun_r', 'isun', 'cifar_cross']
            
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
                root="../data/dtd/images",
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
                root='../data/svhn/', split="test",
                transform=test_transform, download=False
            )
            self.test_loaders['svhn'] = torch.utils.data.DataLoader(
                svhn_data, batch_size=self.args.test_bs, shuffle=True,
                num_workers=4, pin_memory=False
            )
        except:
            pass
        
        # Places365
        try:
            places365_data = dset.ImageFolder(
                root="../data/places365_standard/",
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
                root="../data/LSUN",
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
                root="../data/LSUN_resize",
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
        
        # iSUN
        try:
            isun_data = dset.ImageFolder(
                root="../data/iSUN", transform=test_transform
            )
            self.test_loaders['isun'] = torch.utils.data.DataLoader(
                isun_data, batch_size=self.args.test_bs, shuffle=True,
                num_workers=4, pin_memory=False
            )
        except:
            pass
    
    def train_and_test(self):
        """Complete DAL training and testing pipeline."""
        print(f"Starting DAL training for {self.args.epochs} epochs...")
        print(f"Adaptive gamma: initial={self.gamma}, current={self.current_gamma.item():.3f}")
        
        # Setup test loaders
        if not hasattr(self, 'test_loaders') or not self.test_loaders:
            self.setup_test_loaders()
        
        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            
            # Progress reporting
            if self.args.verbose or epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, "
                      f"Adaptive γ = {self.current_gamma.item():.3f}")
            
            # Periodic accuracy check
            if epoch % 20 == 0 or epoch == self.args.epochs:
                acc = self.test_accuracy()
                print(f"Epoch {epoch}: ID Accuracy = {acc:.2f}%")
        
        print("DAL training completed. Starting evaluation...")
        return self.test()
    
    def get_method_info(self):
        """Get method-specific information."""
        return {
            'method': 'DAL',
            'gamma': self.gamma,
            'beta': self.beta,
            'rho': self.rho,
            'strength': self.strength,
            'iter': self.iter,
            'auxiliary_data': '80 Million Tiny Images',
            'key_technique': 'Embedding Space Augmentation',
            'adaptive_gamma': True,
            'current_gamma': self.current_gamma.item()
        }