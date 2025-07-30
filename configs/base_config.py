"""
Base configuration for shared hyperparameters and settings.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BaseConfig:
    """Base configuration class with shared parameters."""
    
    # Dataset settings
    dataset: str = 'cifar10'  # cifar10 or cifar100
    data_dir: str = '../data/cifarpy'
    
    # Training hyperparameters
    epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 128
    oe_batch_size: int = 256
    test_bs: int = 200
    
    # Optimizer settings
    momentum: float = 0.9
    weight_decay: float = 0.0005
    
    # Model architecture
    layers: int = 40
    widen_factor: int = 2
    droprate: float = 0.3
    
    # Evaluation settings
    eval_mode: str = 'logits'  # 'logits' or 'softmax'
    test_datasets: List[str] = None
    out_as_pos: bool = False
    num_runs: int = 1
    
    # System settings
    num_workers: int = 4
    seed: int = 1
    gpu: int = 0
    verbose: bool = False
    
    def __post_init__(self):
        """Initialize default test datasets if not provided."""
        if self.test_datasets is None:
            self.test_datasets = ['dtd', 'svhn', 'isun', 'cifar_cross']
    
    def get_num_classes(self):
        """Get number of classes based on dataset."""
        return 10 if self.dataset.lower() == 'cifar10' else 100
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'dataset': self.dataset,
            'data_dir': self.data_dir,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'oe_batch_size': self.oe_batch_size,
            'test_bs': self.test_bs,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'layers': self.layers,
            'widen_factor': self.widen_factor,
            'droprate': self.droprate,
            'eval_mode': self.eval_mode,
            'test_datasets': self.test_datasets,
            'out_as_pos': self.out_as_pos,
            'num_runs': self.num_runs,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'gpu': self.gpu,
            'verbose': self.verbose
        }
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments."""
        return cls(
            dataset=getattr(args, 'dataset', 'cifar10'),
            epochs=getattr(args, 'epochs', 100),
            learning_rate=getattr(args, 'learning_rate', 0.01),
            batch_size=getattr(args, 'batch_size', 128),
            oe_batch_size=getattr(args, 'oe_batch_size', 256),
            test_bs=getattr(args, 'test_bs', 200),
            momentum=getattr(args, 'momentum', 0.9),
            weight_decay=getattr(args, 'weight_decay', 0.0005),
            layers=getattr(args, 'layers', 40),
            widen_factor=getattr(args, 'widen_factor', 2),
            droprate=getattr(args, 'droprate', 0.3),
            eval_mode=getattr(args, 'eval_mode', 'logits'),
            test_datasets=getattr(args, 'test_datasets', ['dtd', 'svhn', 'isun', 'cifar_cross']),
            out_as_pos=getattr(args, 'out_as_pos', False),
            num_runs=getattr(args, 'num_runs', 1),
            num_workers=getattr(args, 'num_workers', 4),
            seed=getattr(args, 'seed', 1),
            gpu=getattr(args, 'gpu', 0),
            verbose=getattr(args, 'verbose', False)
        )