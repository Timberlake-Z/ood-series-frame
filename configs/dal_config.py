"""
DAL specific configuration.
"""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass  
class DALConfig(BaseConfig):
    """DAL specific configuration."""
    
    # DAL specific parameters
    gamma: float = 10.0      # Distribution regularization strength
    beta: float = 0.01       # Adaptive adjustment rate  
    rho: float = 10.0        # Target regularization level
    strength: float = 1.0    # Perturbation step size
    iter: int = 10           # Inner iteration steps
    warmup: int = 0          # DAL doesn't use warmup by default
    
    # DAL typically uses more epochs and higher learning rate
    epochs: int = 50
    learning_rate: float = 0.07
    
    # DAL specific settings
    auxiliary_data: str = '80m_tinyimages'
    
    def __post_init__(self):
        """Initialize DAL specific defaults."""
        super().__post_init__()
        
        # Adjust beta based on dataset
        if self.dataset.lower() == 'cifar100' and self.beta == 0.01:
            self.beta = 0.005
    
    def to_dict(self):
        """Convert DAL config to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'gamma': self.gamma,
            'beta': self.beta,
            'rho': self.rho,
            'strength': self.strength,
            'iter': self.iter,
            'warmup': self.warmup,
            'auxiliary_data': self.auxiliary_data
        })
        return base_dict
    
    @classmethod
    def from_args(cls, args):
        """Create DAL config from command line arguments."""
        config = cls()
        
        # Update from base config
        base_config = BaseConfig.from_args(args)
        for key, value in base_config.to_dict().items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # DAL specific parameters
        config.gamma = getattr(args, 'gamma', 10.0)
        config.beta = getattr(args, 'beta', None)
        config.rho = getattr(args, 'rho', 10.0)
        config.strength = getattr(args, 'strength', 1.0)
        config.iter = getattr(args, 'iter', 10)
        config.warmup = getattr(args, 'warmup', 0)
        
        # Auto-adjust beta if not specified
        if config.beta is None:
            config.beta = 0.005 if config.dataset.lower() == 'cifar100' else 0.01
        
        # Override learning rate if not explicitly set
        if not hasattr(args, 'learning_rate') or args.learning_rate == 0.01:
            config.learning_rate = 0.07
            
        # Override epochs if not explicitly set  
        if not hasattr(args, 'epochs') or args.epochs == 100:
            config.epochs = 50
        
        return config
    
    @classmethod
    def get_recommended_configs(cls):
        """Get recommended configurations for different datasets."""
        configs = {}
        
        # CIFAR-10 configuration
        configs['cifar10'] = cls(
            dataset='cifar10',
            gamma=10.0,
            beta=0.01,
            rho=10.0,
            strength=1.0,
            iter=10,
            epochs=50,
            learning_rate=0.07
        )
        
        # CIFAR-100 configuration
        configs['cifar100'] = cls(
            dataset='cifar100', 
            gamma=10.0,
            beta=0.005,  # Lower beta for CIFAR-100
            rho=10.0,
            strength=1.0,
            iter=10,
            epochs=50,
            learning_rate=0.07
        )
        
        return configs