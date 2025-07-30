# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

回复问题使用中文
代码和注释使用英文

## Overview

This repository contains two related research projects for out-of-distribution (OOD) detection:

### 1. W-DOE (Wasserstein Distribution-agnostic Outlier Exposure) - IEEE TPAMI 2025
- **Location**: Root directory
- **Entry point**: `wdoe_final.py`
- **Method**: Implicit data synthesis using Adversarial Weight Perturbation
- **Auxiliary OOD data**: TinyImageNet-200

### 2. DAL (Distributional-Augmented OOD Learning) - NeurIPS 2023  
- **Location**: `DAL-main/` directory
- **Entry point**: `main.py`
- **Method**: Wasserstein ball-based distribution augmentation
- **Auxiliary OOD data**: 80 Million Tiny Images

Both projects focus on improving OOD detection by addressing the distribution gap between auxiliary OOD training data and real-world unseen OOD data. W-DOE is the more recent and advanced method.

## Training Commands

### W-DOE Training (Root Directory)
```bash
python wdoe_final.py {dataset_name} --gamma={hyper_value} --warmup={warmup_value} --epochs={epoch_value} --learning_rate={learning_rate}
```

Where:
- `dataset_name`: Choose from `cifar10` or `cifar100`
- Key hyperparameters: `--gamma` (W-DOE specific), `--warmup`, `--epochs`, `--learning_rate`
- Example: `python wdoe_final.py cifar10 --gamma=0.5 --warmup=5 --epochs=100 --learning_rate=0.01`

### DAL Training (DAL-main Directory)
```bash
cd DAL-main
python main.py cifar10 --gamma=10 --beta=.01 --rho=10 --iter=10 --learning_rate=0.07 --strength=1
python main.py cifar100 --gamma=10 --beta=.005 --rho=10 --iter=10 --learning_rate=0.07 --strength=1
```

### Test-only Mode
```bash
# W-DOE
python wdoe_final.py {dataset} --test

# DAL  
cd DAL-main && python main.py {dataset} --test
```

## Architecture

### W-DOE Components (Root Directory)

**Main Training Script**: `wdoe_final.py`
- W-DOE implementation with implicit data synthesis (IDS)
- Uses Adversarial Weight Perturbation for data augmentation

**AWP Utilities**: `utils/utils_awp.py`
- Adversarial Weight Perturbation functionality
- Key functions: `diff_in_weights()`, `add_into_weights()`, `average_diff()`
- Core component for W-DOE's implicit data synthesis

**Validation Split**: `utils/validation_dataset.py`
- `PartialDataset` class for dataset splitting
- `validation_split()` function for train/val separation

### DAL Components (DAL-main Directory)

**Main Training Script**: `DAL-main/main.py`
- DAL implementation with Wasserstein ball-based augmentation
- Uses 80 Million Tiny Images for auxiliary OOD data

**TinyImages Loader**: `DAL-main/utils/tinyimages_80mn_loader.py`
- Handles 80 Million Tiny Images dataset loading
- Provides auxiliary OOD data for DAL training

### Shared Components

**Model Architecture**: `models/wrn.py` (both directories)
- Wide ResNet implementation with configurable depth/width
- Standard architecture: 40 layers, widen factor of 2, dropout 0.3
- Implements BasicBlock and NetworkBlock components

**Evaluation**: `utils/display_results.py` (both directories)
- OOD detection metrics: AUROC, AUPR, FPR@95
- Functions: `get_measures()`, `print_measures()`
- Comprehensive evaluation utilities for OOD detection benchmarks

**SVHN Loader**: `utils/svhn_loader.py` (both directories)
- Custom SVHN dataset implementation for auxiliary OOD data
- Supports train/test/extra splits

## Dependencies

- Python 3.7.10
- PyTorch 1.7.1
- torchvision 0.8.2
- CUDA support required
- NumPy, scikit-learn

## Data Structure and Key Differences

### Auxiliary OOD Data (训练时使用的辅助OOD数据)

**W-DOE使用TinyImageNet-200**:
- 路径: `../data/tiny-imagenet-200/train/`
- 数据加载: `dset.ImageFolder` with resize/crop transforms
- 特点: 相对较小的数据集，200个类别

**DAL使用80 Million Tiny Images**:
- 路径: `/data/csqzwang/tiny_images.bin` (binary file)
- 数据加载: 自定义`TinyImages`类，读取二进制文件
- 特点: 超大规模数据集(7930万张图片)，自动排除CIFAR重叠图片
- 索引文件: `./DAL-main/utils/80mn_cifar_idxs.txt`

### Test OOD Datasets (测试时用的OOD数据集)

**W-DOE**: 只使用部分测试集(代码中注释掉了places365/LSUN)
- Textures (DTD), iSUN, SVHN, CIFAR互测

**DAL**: 使用完整的测试集
- Textures, Places365, LSUN-C, LSUN-R, iSUN, SVHN, CIFAR互测

### 共用数据集的处理差异

#### 1. CIFAR数据集支持差异
- **W-DOE**: 只实现了CIFAR-10支持 (CIFAR-100代码被注释)
- **DAL**: 完整支持CIFAR-10和CIFAR-100

#### 2. 数据预处理完全一致
- **标准化参数**: 两个项目都使用相同的CIFAR归一化参数
  ```python
  mean = [0.4914, 0.4822, 0.4465]  # [125.3/255, 123.0/255, 113.9/255]
  std = [0.2471, 0.2435, 0.2616]   # [63.0/255, 62.1/255, 66.7/255]
  ```
- **Transform管道**: 训练和测试变换完全相同

#### 3. 数据加载器配置差异
- **W-DOE**: 
  - 训练loader: `num_workers=args.prefetch` (可变)
  - 测试loader: `num_workers=4` (固定)
  - SVHN: `download=True`
- **DAL**: 
  - 所有loader: `num_workers=4` (统一)
  - SVHN: `download=False`

#### 4. 测试数据集路径差异
- **W-DOE wdoe_tim.py**: DTD路径为 `"../doe/data/dtd/images"` (可能是路径错误)
- **W-DOE wdoe_final.py & DAL**: DTD路径为 `"../data/dtd/images"` (标准路径)

Expected data directory structure:
```
../data/
├── tiny-imagenet-200/     # W-DOE auxiliary OOD data
├── 80mn_cifar_idxs.txt    # DAL CIFAR exclusion indices  
├── dtd/images/            # Textures test set
├── places365_standard/    # Places365 test set (DAL only)
├── LSUN/                  # LSUN test sets (DAL only)
├── LSUN_resize/           # LSUN resized (DAL only)  
└── iSUN/                  # iSUN test set
```

Pretrained models:
- **W-DOE**: `./ckpt/cifar{10,100}_wrn_pretrained_epoch_99.pt`
- **DAL**: `./DAL-main/models/cifar{10,100}_wrn_pretrained_epoch_99.pt`

## Training Workflow Differences

### W-DOE Training Process
1. **两阶段训练策略**:
   - Warmup阶段 (epoch < warmup): 同时优化分类损失和OE损失 `loss = l_ce + l_oe`
   - 主训练阶段 (epoch >= warmup): 只优化OE损失 `loss = l_oe`

2. **Implicit Data Synthesis (核心创新)**:
   - 创建proxy网络，加载主网络权重
   - 计算AWP perturbation: `diff = diff_in_weights(net, proxy)`
   - 应用perturbation到主网络: `add_into_weights(net, diff, coeff=gamma)`
   - 通过权重扰动实现隐式数据变换

3. **损失函数**:
   - `l_ce`: 标准交叉熵损失(ID数据)
   - `l_oe`: Outlier Exposure损失(OOD数据)
   - `l_reg`: 特征正则化损失(控制perturbation强度)

### DAL Training Process  
1. **端到端训练**:
   - 固定损失组合: `loss = l_ce + 0.5 * l_oe`
   - 无warmup机制，全程同时优化两个损失

2. **Embedding Augmentation (核心创新)**:
   - 在特征空间进行对抗性增强
   - 迭代优化embedding bias: `emb_bias = emb_bias + strength * grads`
   - 使用增强的embedding计算OE损失: `x_oe = net.fc(emb + emb_bias)`

3. **自适应gamma调节**:
   - 动态调整正则化参数: `gamma -= beta * (rho - r_sur)`
   - 基于当前正则化强度自适应调节

### 评估方式差异
- **评分计算**: W-DOE使用原始logits，DAL使用softmax概率
- **测试集**: W-DOE测试集较少，DAL包含完整基准测试集
- **重复实验**: W-DOE支持多次运行平均，DAL单次运行

## Key Implementation Details

### W-DOE (Parameter-level Augmentation)
- **Method**: Adversarial Weight Perturbation (AWP)在参数空间增强
- **Data**: TinyImageNet-200 (小规模，高质量)
- **Training**: 两阶段策略 + warmup机制
- **Innovation**: 权重扰动→隐式数据变换

### DAL (Feature-level Augmentation)  
- **Method**: Embedding augmentation在特征空间增强
- **Data**: 80 Million Tiny Images (大规模，覆盖广)
- **Training**: 端到端优化 + 自适应参数调节
- **Innovation**: 特征空间对抗增强 + Wasserstein ball约束

### 方法论对比
- **W-DOE**: 从参数扰动角度解决分布gap，更加理论化
- **DAL**: 从特征增强角度解决分布gap，更加直观和工程化
- **共同目标**: 都旨在通过增强辅助OOD数据来缩小与真实OOD数据的分布差距