# W-DOE: Wasserstein Distribution-agnostic Outlier Exposure

**[W-DOE: Wasserstein Distribution-agnostic Outlier Exposure](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10844561)**   (IEEE Transactions on Pattern Analysis and Machine Intelligence 2025)

Qizhou Wang, Bo Han, Yang Liu, Chen Gong, Tongliang Liu, Jiming Liu. 




**Keywords**: Out-of-distribution Detection, Reliable Machine Learning


**Abstract**: In open-world environments, classification models should be adept at identifying out-of-distribution (OOD) data whose semantics differ from in-distribution (ID) data, leading to the emerging research in OOD detection. As a promising learning scheme, outlier exposure (OE) enables the models to learn from auxiliary OOD data, enhancing model representations in discerning between ID and OOD patterns. However, these auxiliary OOD data often do not fully represent real OOD scenarios, potentially biasing our models in practical OOD detection. Hence, we propose a novel OE-based learning method termed Wasserstein Distribution-agnostic Outlier Exposure (W-DOE), which is both theoretically sound and experimentally superior to previous works. The intuition is that by expanding the coverage of training-time OOD data, the models will encounter fewer unseen OOD cases upon deployment. In W-DOE, we achieve additional OOD data to enlarge the OOD coverage, based on a new data synthesis approach called implicit data synthesis (IDS). It is driven by our new insight that perturbing model parameters can lead to implicit data transformation, which is simple to implement yet effective to realize. Furthermore, we suggest a general learning framework to search for the synthesized OOD data that can benefit the models most, ensuring the OOD performance for the enlarged OOD coverage measured by the Wasserstein metric. Our approach comes with provable guarantees for open-world settings, demonstrating that broader OOD coverage ensures reduced estimation errors and thereby improved generalization for real OOD cases. We conduct extensive experiments across a series of representative OOD detection setups, further validating the superiority of W-DOE against state-of-the-art counterparts in the field. 

```
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
```

## Get Started

### Environment
- Python (3.7.10)
- Pytorch (1.7.1)
- torchvision (0.8.2)
- CUDA
- Numpy

### Pretrained Models and Datasets

Pretrained models are provided in folder

```
./ckpt/
```

Please download the datasets in folder

```
../data/
```

Surrogate OOD Dataset

- [tiny-ImageNet-200](https://github.com/chihhuiho/CLAE/blob/main/datasets/download_tinyImagenet.sh)


Test OOD Datasets 

- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

- [Places365](http://places2.csail.mit.edu/download.html)

- [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
 
- [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)


### File Structure

After the preparation work, the whole project should have the following structure:

```
./DOE
├── README.md
├── ckpt                            # datasets
│   ├── cifar10_wrn_pretrained_epoch_99.pt 
│   └── cifar100_wrn_pretrained_epoch_99.pt
├── models                          # models
│   └── wrn.py
├── utils                           # utils
│   ├── display_results.py                        
│   ├── utils_awp.py
│   └── validation_dataset.py
└── doe_final.py                    # training code
```



## Training

To train the DOE model on CIFAR benckmarks, simply run:

- CIFAR-10
```train cifar10
python doe_final.py cifar10 
```


- CIFAR-100
```train cifar100
python doe_final.py cifar100
```

## Results

The key results on CIFAR benchmarks are listed in the following table. 

|     | CIFAR-10 | CIFAR-10 | CIFAR-100 | CIFAR-100 |
|:---:|:--------:|:--------:|:---------:|:---------:|
|     |   FPR95  |   AUROC  |   FPR95   |   AUROC   |
| MSP |   53.77  |   88.40  |   76.73   |   76.24   |
|  OE |   12.41  |   97.85  |   45.68   |   87.61   |
| DOE |   **5.15**   |   **98.78**  |   **25.38**   |   **93.97**   |
