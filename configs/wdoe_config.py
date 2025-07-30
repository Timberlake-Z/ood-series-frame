"""
W-DOE specific configuration.
"""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class WDOEConfig(BaseConfig):
    """W-DOE specific configuration."""
    
    # W-DOE specific parameters
    gamma: float = 0.5  # Regularization strength
    warmup: int = 5     # Warmup epochs
    
    # W-DOE typically uses fewer epochs
    epochs: int = 10
    
    # W-DOE specific settings
    auxiliary_data: str = 'tinyimagenet200'
    
    def __post_init__(self):
        """Initialize W-DOE specific defaults."""
        super().__post_init__()
        
        # W-DOE uses all available test datasets
        if self.test_datasets == ['dtd', 'svhn', 'isun', 'cifar_cross']:
            self.test_datasets = ['dtd', 'svhn', 'isun', 'cifar_cross', 'places365', 'lsun_c', 'lsun_r']
    
    def to_dict(self):
        """Convert W-DOE config to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'gamma': self.gamma,
            'warmup': self.warmup,
            'auxiliary_data': self.auxiliary_data
        })
        return base_dict
    
    @classmethod  
    def from_args(cls, args):
        """Create W-DOE config from command line arguments."""
        config = cls()
        
        # Update from base config
        base_config = BaseConfig.from_args(args)
        for key, value in base_config.to_dict().items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # W-DOE specific parameters
        config.gamma = getattr(args, 'gamma', 0.5)
        config.warmup = getattr(args, 'warmup', 5)
        
        # Override epochs if not explicitly set
        if not hasattr(args, 'epochs') or args.epochs == 100:
            config.epochs = 10
        
        return config
    
    @classmethod
    def get_recommended_configs(cls):
        """Get recommended configurations for different datasets."""
        configs = {}
        
        # CIFAR-10 configuration
        configs['cifar10'] = cls(
            dataset='cifar10',
            gamma=0.5,
            warmup=5,
            epochs=10,
            learning_rate=0.01
        )
        
        # CIFAR-100 configuration  
        configs['cifar100'] = cls(
            dataset='cifar100',
            gamma=0.5,
            warmup=5,
            epochs=10,
            learning_rate=0.01
        )
        
        return configs