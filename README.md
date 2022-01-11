# Continual learning datasets

## Introduction

<p align="justify"> This repository contains PyTorch image dataloaders and utility functions to load datasets for supervised continual learning. Currently supported datasets:

- MNIST
- Pairwise-MNIST
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- not-MNIST (letters version of MNIST, see [EMNIST](https://github.com/hosford42/EMNIST) for more detail)
- CIFAR-10
- CIFAR-100
- [German Traffic Signs](https://benchmark.ini.rub.de/) 
- Street View House Numbers (SVHN)
- Incremental CIFAR-100
- Incremental TinyImageNet
</p>

## Features
<p align="justify"> 
The provided interface simplifies typical data loading for supervised continual learning scenarios.

- Dataset order, additional training data (for replay buffers) and test data (for global metrics computation) can all be specified.

- A *batch balancing* feature is also available to make sure data from all available classes are available in a training batch.</p>

- Training data size and channels can be specified. Transformations will be added to make sure input data always has the same size and number of channels. If a single channel is specified, grayscaling will be applied. Otherwise, if 3 channels are specified, single channels will be triplicated. Bicubic interpolation or linear subsampling will be applied to meet the specified size.

# Installation

1. Clone the repository to your machine.
2. Install the package:
```bash
pip install -e cl_datasets/
```

**Note**: Please use Python 3.8 or above.

# Example

```python
from cl_datasets import getDatasets

datasets = ['svhn','cifar10','fashion','mnist']
batchSize = 32
dataSize = (32,32)
nChannels = 3

dataloaders = getDatasets(datasets,batchSize,dataSize,nChannels)

for train_test_loaders in dataloaders:
    trainLoader,testLoader = train_test_loaders
    ...
```

# List of possible datasets for training tasks

## Full datasets

| Description          |           Dataset string |
| -------------------- | -----------------------: |
| MNIST                |       "mnist" or "MNIST" |
| not-MNIST            | "notMnist" or "notMNIST" |
| Fashion MNIST        |                "fashion" |
| SVHN                 |                   "svhn" |
| Cifar-10             |                "cifar10" |
| Cifar-100            |               "cifar100" |
| German traffic signs |                "traffic" |

## Incremental datasets

| Description                                     |                   Dataset string |
| ----------------------------------------------- | -------------------------------: |
| Pairwise MNIST                                  |     "mnist_xy" (e.g. "mnist_01") |
| Incremental Cifar-100 (10 classes per task)     | "cifar100_i" (e.g. "cifar100_4") |
| Incremental Tiny ImageNet (10 classes per task) |           "TIN_i" (e.g. "TIN_3") |

