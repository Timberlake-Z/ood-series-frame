# Unified OOD Detection Framework

A unified framework integrating **W-DOE** (Wasserstein Distribution-agnostic Outlier Exposure) and **DAL** (Distributional-Augmented OOD Learning) methods for out-of-distribution detection.

## Overview

This framework provides a single codebase to train, test, and compare two state-of-the-art OOD detection methods:

- **W-DOE (IEEE TPAMI 2025)**: Implicit data synthesis using Adversarial Weight Perturbation
- **DAL (NeurIPS 2023)**: Wasserstein ball-based distribution augmentation

## Key Features

- 🚀 **Unified Interface**: Single command-line entry point for both methods
- 🔧 **Flexible Configuration**: Multiple ways to configure hyperparameters
- 📊 **Comprehensive Evaluation**: Standard OOD detection benchmarks
- 🔄 **Easy Comparison**: Built-in method comparison functionality
- 📁 **Organized Structure**: Modular design for easy maintenance and extension

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install torch torchvision
```

### Basic Usage

```bash
# Train W-DOE on CIFAR-10
python unified_main.py --method wdoe --dataset cifar10 --epochs 10

# Train DAL on CIFAR-100  
python unified_main.py --method dal --dataset cifar100 --epochs 50

# Compare both methods
python unified_main.py --method both --dataset cifar10
```

### Test Only Mode

```bash
# W-DOE testing
python unified_main.py --method wdoe --dataset cifar10 --test

# DAL testing
python unified_main.py --method dal --dataset cifar100 --test
```

## Framework Architecture

```
├── unified_main.py           # Main entry point
├── methods/                  # Training method implementations
│   ├── base_trainer.py       # Base trainer class
│   ├── wdoe_trainer.py       # W-DOE implementation
│   └── dal_trainer.py        # DAL implementation
├── data/                     # Data loading modules
├── configs/                  # Configuration management
├── models/                   # Model definitions
└── utils/                    # Utility functions
```

## Methods

### W-DOE (Wasserstein Distribution-agnostic Outlier Exposure)
- **Paper**: [IEEE TPAMI 2025](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10844561)
- **Key Innovation**: Implicit data synthesis via Adversarial Weight Perturbation
- **Auxiliary Data**: TinyImageNet-200
- **Training Strategy**: Two-stage with warmup mechanism

### DAL (Distributional-Augmented OOD Learning)  
- **Paper**: NeurIPS 2023
- **Key Innovation**: Wasserstein ball-based distribution augmentation
- **Auxiliary Data**: 80 Million Tiny Images
- **Training Strategy**: End-to-end with adaptive parameter adjustment

## Configuration

The framework supports multiple configuration methods:

1. **Command Line Arguments** (highest priority)
2. **Configuration Files** (YAML format)
3. **Default Configuration Classes**

See [使用说明.md](使用说明.md) for detailed configuration options.

## Data Setup

### Automatic Downloads
- CIFAR-10/100: Automatically downloaded
- SVHN: Automatically downloaded

### Manual Setup Required
```bash
# TinyImageNet-200 (for W-DOE)
cd ../data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

# DTD (Describable Textures Dataset)
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz
```

See [使用说明.md](使用说明.md) for complete data setup instructions.

## Results

The framework evaluates on standard OOD detection benchmarks:

**Test OOD Datasets:**
- Textures (DTD)
- SVHN  
- iSUN
- Places365 (DAL only)
- LSUN (DAL only)

**Metrics:**
- AUROC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)  
- FPR95 (False Positive Rate at 95% TPR)

## Hyperparameters

### W-DOE Default Settings
- `gamma`: 0.5 (AWP regularization strength)
- `warmup`: 5 epochs
- `learning_rate`: 0.01
- `epochs`: 10

### DAL Default Settings  
- `gamma`: 10.0 (distribution regularization)
- `beta`: 0.01 (adaptive adjustment rate)
- `rho`: 10.0 (target regularization level)
- `learning_rate`: 0.07
- `epochs`: 50

## Documentation

- [使用说明.md](使用说明.md) - Complete usage guide (Chinese)
- [CLAUDE.md](CLAUDE.md) - Technical documentation for Claude Code
- [docs/](docs/) - Detailed analysis and implementation reports

## Citation

If you use this framework in your research, please cite the original papers:

```bibtex
@ARTICLE{wang2025wdoe,
author={Wang, Qizhou and Han, Bo and Liu, Yang and Gong, Chen and Liu, Tongliang and Liu, Jiming},
journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
title={W-DOE: Wasserstein Distribution-agnostic Outlier Exposure},
year={2025},
number={01},
pages={1-14},
doi={10.1109/TPAMI.2025.3531000},
url={https://doi.ieeecomputersociety.org/10.1109/TPAMI.2025.3531000},
publisher={IEEE},
}

@inproceedings{dal2023,
  title={Distributional-Augmented OOD Learning},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original W-DOE implementation by Qizhou Wang et al.
- Original DAL implementation  
- PyTorch community for the excellent framework
- Claude Code for framework integration assistance
