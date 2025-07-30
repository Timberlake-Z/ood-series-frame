#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Training Framework for W-DOE and DAL Methods

This script provides a unified interface to train and evaluate both W-DOE and DAL methods
for out-of-distribution detection, enabling fair comparison under consistent settings.

Usage:
    python unified_main.py --method wdoe --dataset cifar10 --gamma 0.5 --warmup 5
    python unified_main.py --method dal --dataset cifar10 --gamma 10 --beta 0.01
    python unified_main.py --method both --dataset cifar10 --compare
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def create_parser():
    """Create unified argument parser for both methods."""
    parser = argparse.ArgumentParser(
        description='Unified Training Framework for W-DOE and DAL Methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument('--method', type=str, required=True,
                       choices=['wdoe', 'dal', 'both'],
                       help='Method to use: wdoe, dal, or both for comparison')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use: cifar10 or cifar100')
    
    # Experimental control
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison between methods (only with --method both)')
    parser.add_argument('--test_only', '-t', action='store_true',
                       help='Test only, skip training')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of experimental runs for averaging')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed for reproducibility')
    
    # Training hyperparameters (shared)
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                       help='Batch size for ID data')
    parser.add_argument('--oe_batch_size', type=int, default=256,
                       help='Batch size for OOD auxiliary data')
    parser.add_argument('--test_bs', type=int, default=200,
                       help='Batch size for testing')
    
    # Optimizer settings
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', '-d', type=float, default=0.0005,
                       help='Weight decay (L2 penalty)')
    
    # Model architecture
    parser.add_argument('--layers', default=40, type=int,
                       help='Number of layers in WideResNet')
    parser.add_argument('--widen_factor', default=2, type=int,
                       help='Widen factor for WideResNet')
    parser.add_argument('--droprate', default=0.3, type=float,
                       help='Dropout probability')
    
    # W-DOE specific parameters
    parser.add_argument('--gamma', type=float, default=None,
                       help='Regularization strength (method-specific defaults apply)')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Warmup epochs for W-DOE')
    
    # DAL specific parameters  
    parser.add_argument('--beta', type=float, default=None,
                       help='DAL beta parameter (adaptive adjustment rate)')
    parser.add_argument('--rho', type=float, default=None,
                       help='DAL rho parameter (target regularization)')
    parser.add_argument('--strength', type=float, default=None,
                       help='DAL strength parameter (perturbation step size)')
    parser.add_argument('--iter', type=int, default=10,
                       help='DAL inner iteration steps')
    
    # Evaluation settings
    parser.add_argument('--eval_mode', type=str, default='logits',
                       choices=['logits', 'softmax'],
                       help='Evaluation scoring mode for fair comparison')
    parser.add_argument('--test_datasets', nargs='+',
                       default=['dtd', 'svhn', 'isun', 'cifar_cross'],
                       help='OOD test datasets to evaluate')
    parser.add_argument('--out_as_pos', action='store_true',
                       help='Treat OOD as positive class in evaluation')
    
    # System settings
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser

def setup_environment(args):
    """Setup experimental environment."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("CUDA not available, using CPU")
    
    # Set cuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_method_defaults(args):
    """Apply method-specific default parameters."""
    if args.method == 'wdoe':
        # W-DOE defaults
        if args.gamma is None:
            args.gamma = 0.5
        if args.epochs == 100:  # If using default
            args.epochs = 10  # W-DOE typically uses fewer epochs
    
    elif args.method == 'dal':
        # DAL defaults
        if args.gamma is None:
            args.gamma = 10.0
        if args.beta is None:
            args.beta = 0.01 if args.dataset == 'cifar10' else 0.005
        if args.rho is None:
            args.rho = 10.0
        if args.strength is None:
            args.strength = 1.0
        if args.learning_rate == 0.01:  # If using default
            args.learning_rate = 0.07  # DAL uses higher LR
    
    elif args.method == 'both':
        # For comparison, we need to create configs for both methods
        print("Comparison mode: Will run both methods with their respective defaults")

def validate_args(args):
    """Validate argument combinations."""
    if args.method == 'both' and not args.compare:
        print("Warning: Using --method both without --compare. Adding --compare automatically.")
        args.compare = True
    
    if args.compare and args.method != 'both':
        raise ValueError("--compare can only be used with --method both")
    
    if args.method == 'dal' and args.dataset == 'cifar100':
        # Adjust DAL parameters for CIFAR-100 if not specified
        if args.beta is None:
            args.beta = 0.005

def run_single_method(args, method_name=None):
    """Run training/evaluation for a single method."""
    if method_name:
        current_method = method_name
    else:
        current_method = args.method
    
    print(f"\n{'='*60}")
    print(f"Running {current_method.upper()} on {args.dataset.upper()}")
    print(f"{'='*60}")
    
    try:
        if current_method == 'wdoe':
            from methods.wdoe_trainer import WDOETrainer
            trainer = WDOETrainer(args)
        elif current_method == 'dal':
            from methods.dal_trainer import DALTrainer  
            trainer = DALTrainer(args)
        else:
            raise ValueError(f"Unknown method: {current_method}")
        
        # Run training or testing
        if args.test_only:
            results = trainer.test()
        else:
            results = trainer.train_and_test()
        
        return results
        
    except ImportError as e:
        print(f"Error importing {current_method} trainer: {e}")
        print("Please ensure the methods module is properly set up.")
        return None
    except Exception as e:
        print(f"Error running {current_method}: {e}")
        return None

def run_comparison(args):
    """Run comparison between W-DOE and DAL methods."""
    print(f"\n{'='*60}")
    print(f"COMPARISON MODE: W-DOE vs DAL on {args.dataset.upper()}")
    print(f"{'='*60}")
    
    results = {}
    
    # Run W-DOE
    args_wdoe = argparse.Namespace(**vars(args))
    args_wdoe.method = 'wdoe'
    apply_method_defaults(args_wdoe)
    print(f"\n[1/2] Running W-DOE...")
    results['wdoe'] = run_single_method(args_wdoe)
    
    # Run DAL
    args_dal = argparse.Namespace(**vars(args))
    args_dal.method = 'dal'
    apply_method_defaults(args_dal)
    print(f"\n[2/2] Running DAL...")
    results['dal'] = run_single_method(args_dal)
    
    # Print comparison results
    print_comparison_results(results, args.dataset)
    
    return results

def print_comparison_results(results, dataset):
    """Print formatted comparison results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS ON {dataset.upper()}")
    print(f"{'='*80}")
    
    if results['wdoe'] is None or results['dal'] is None:
        print("Error: Some methods failed to run. Cannot show comparison.")
        return
    
    # Print header
    print(f"{'Dataset':<12} {'Method':<8} {'FPR95':<8} {'AUROC':<8} {'AUPR':<8}")
    print("-" * 50)
    
    # Print results for each test dataset
    for test_dataset in results['wdoe'].keys():
        if test_dataset in results['dal']:
            wdoe_res = results['wdoe'][test_dataset]
            dal_res = results['dal'][test_dataset]
            
            print(f"{test_dataset:<12} {'W-DOE':<8} {wdoe_res[0]:<8.2f} {wdoe_res[1]:<8.2f} {wdoe_res[2]:<8.2f}")
            print(f"{'':<12} {'DAL':<8} {dal_res[0]:<8.2f} {dal_res[1]:<8.2f} {dal_res[2]:<8.2f}")
            print("-" * 50)

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate and setup
    validate_args(args)
    setup_environment(args)
    apply_method_defaults(args)
    
    if args.verbose:
        print("Arguments:")
        for key, value in sorted(vars(args).items()):
            print(f"  {key}: {value}")
        print()
    
    # Run experiments
    try:
        if args.compare:
            results = run_comparison(args)
        else:
            results = run_single_method(args)
        
        if results is not None:
            print("\nExperiment completed successfully!")
        else:
            print("\nExperiment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()