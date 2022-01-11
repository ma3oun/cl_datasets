"""
Datasets loader
"""

from typing import Union, Iterable
import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from cl_datasets.tinyImageNet import TinyImageNet
from cl_datasets.trafficSigns import TrafficSigns
from cl_datasets.notMNIST import NotMNIST
from cl_datasets.sampling import BalancedBatchSampler


def _getRoot(rootDir: str = None) -> str:
    if rootDir is None:
        root = os.environ.get("DATASETS_ROOT")
        if root is None:
            root = "_data"
    else:
        root = rootDir
    return root


def _sizesAreEqual(
    newSize: Union[int, Iterable], originalSize: Union[int, Iterable]
) -> bool:
    """
    Compare image shapes.

    Args:

        newSize: First size operand
        originalSize: Second size operand
    
    Return:
        True if sizes are equal
    """

    def toTupleSize(size):
        if type(size) is int:
            tupleSize = (size, size)
        else:
            tupleSize = tuple(size)
        return tupleSize

    _newSize = toTupleSize(newSize)
    _originalSize = toTupleSize(originalSize)

    return _newSize == _originalSize


def getNotMNIST(
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:

    originalSize = (28, 28)

    if size is None or _sizesAreEqual(size[-2:], originalSize[-2:]):
        tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [resize, transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    if channels == 3:
        tfms = transforms.Compose(
            [tfms, transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

    trainDataset = (
        NotMNIST(
            _getRoot(dataDir),
            train=True,
            download=True,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTrainingData
    )
    testDataset = (
        NotMNIST(
            _getRoot(dataDir),
            train=False,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTestData
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(trainDataset.labels, 10, batchSize)
        testBatchSampler = BalancedBatchSampler(testDataset.labels, 10, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_sampler=trainBatchSampler, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_sampler=testBatchSampler, **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )
    return train_loader, test_loader


def getTrafficSigns(
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (32, 32)
    if size is None or _sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669),
                ),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669),
                ),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    trainDataset = (
        TrafficSigns(
            _getRoot(dataDir),
            train=True,
            download=True,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTrainingData
    )
    testDataset = (
        TrafficSigns(
            _getRoot(dataDir),
            train=False,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTestData
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(trainDataset.labels, 43, batchSize)
        testBatchSampler = BalancedBatchSampler(testDataset.labels, 43, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_sampler=trainBatchSampler, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_sampler=testBatchSampler, **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )
    return train_loader, test_loader


def getTinyImageNet(
    batchSize: int,
    taskID: int,
    nTasks: int = 10,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (64, 64)
    nClsPerTask = 200 // nTasks

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if size is None or _sizesAreEqual(size, originalSize):
        tfms = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose([resize, transforms.ToTensor(), normalize])

    if channels == 1:
        tfms = transforms.Compose([transforms.Grayscale(num_output_channels=1), tfms])

    train = TinyImageNet(
        _getRoot(dataDir),
        train=True,
        transform=tfms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    test = TinyImageNet(
        _getRoot(dataDir),
        train=False,
        transform=tfms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    targets_train = torch.tensor(train.targets)
    targets_train_idx = (targets_train >= taskID * nClsPerTask) & (
        targets_train < (taskID + 1) * nClsPerTask
    )

    targets_test = torch.tensor(test.targets)
    targets_test_idx = (targets_test >= taskID * nClsPerTask) & (
        targets_test < (taskID + 1) * nClsPerTask
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(train.targets, nClsPerTask, batchSize)
        testBatchSampler = BalancedBatchSampler(test.targets, nClsPerTask, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0])
            + extraTrainingData,
            batch_sampler=trainBatchSampler,
            **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0])
            + extraTestData,
            batch_sampler=testBatchSampler,
            **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0])
            + extraTrainingData,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
            **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0])
            + extraTestData,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
            **kwargs,
        )

    return train_loader, test_loader


def getPairwiseMNIST(
    batchSize: int,
    labels: tuple,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (28, 28)
    if size is None or _sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [resize, transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    if channels == 3:
        tfms = transforms.Compose(
            [tfms, transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

    train = datasets.MNIST(
        _getRoot(dataDir),
        train=True,
        download=True,
        transform=tfms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    test = datasets.MNIST(
        _getRoot(dataDir),
        train=False,
        transform=tfms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    targets_train = train.targets.clone().detach()
    targets_train_idx = (targets_train == labels[0]) | (targets_train == labels[1])

    targets_test = test.targets.clone().detach()
    targets_test_idx = (targets_test == labels[0]) | (targets_test == labels[1])

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(train.targets, 2, batchSize)
        testBatchSampler = BalancedBatchSampler(test.targets, 2, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0])
            + extraTrainingData,
            batch_sampler=trainBatchSampler,
            **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0])
            + extraTestData,
            batch_sampler=testBatchSampler,
            **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0])
            + extraTrainingData,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
            **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0])
            + extraTestData,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
            **kwargs,
        )

    return train_loader, test_loader


def getMNIST(
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (28, 28)

    if size is None or _sizesAreEqual(size[-2:], originalSize[-2:]):
        tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [resize, transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    if channels == 3:
        tfms = transforms.Compose(
            [tfms, transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

    trainDataset = (
        datasets.MNIST(
            _getRoot(dataDir),
            train=True,
            download=True,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTrainingData
    )
    testDataset = (
        datasets.MNIST(
            _getRoot(dataDir),
            train=False,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTestData
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(trainDataset.targets, 10, batchSize)
        testBatchSampler = BalancedBatchSampler(testDataset.targets, 10, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_sampler=trainBatchSampler, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_sampler=testBatchSampler, **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )
    return train_loader, test_loader


def getFashion(
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (28, 28)
    if size is None or _sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [resize, transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    if channels == 3:
        tfms = transforms.Compose(
            [tfms, transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

    trainDataset = (
        datasets.FashionMNIST(
            _getRoot(dataDir),
            train=True,
            download=True,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTrainingData
    )
    testDataset = (
        datasets.FashionMNIST(
            _getRoot(dataDir),
            train=False,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTestData
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(trainDataset.targets, 10, batchSize)
        testBatchSampler = BalancedBatchSampler(testDataset.targets, 10, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_sampler=trainBatchSampler, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_sampler=testBatchSampler, **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )

    return train_loader, test_loader


def getCifar100(
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (32, 32)
    if size is None or _sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    trainDataset = (
        datasets.CIFAR100(
            _getRoot(dataDir),
            train=True,
            download=True,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTrainingData
    )
    testDataset = (
        datasets.CIFAR100(
            _getRoot(dataDir),
            train=False,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTestData
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(trainDataset.targets, 100, batchSize)
        testBatchSampler = BalancedBatchSampler(testDataset.targets, 100, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_sampler=trainBatchSampler, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_sampler=testBatchSampler, **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )
    return train_loader, test_loader


def getIncrementalCifar100(
    batchSize: int,
    taskID: int,
    nTasks: int = 10,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (32, 32)
    if size is None or _sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    nClsPerTask = 100 // nTasks

    train = datasets.CIFAR100(
        _getRoot(dataDir),
        train=True,
        download=True,
        transform=tfms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    test = datasets.CIFAR100(
        _getRoot(dataDir),
        train=False,
        transform=tfms,
        target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
    )

    targets_train = torch.tensor(train.targets)
    targets_train_idx = (targets_train >= taskID * nClsPerTask) & (
        targets_train < (taskID + 1) * nClsPerTask
    )

    targets_test = torch.tensor(test.targets)
    targets_test_idx = (targets_test >= taskID * nClsPerTask) & (
        targets_test < (taskID + 1) * nClsPerTask
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(train.targets, nClsPerTask, batchSize)
        testBatchSampler = BalancedBatchSampler(test.targets, nClsPerTask, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0])
            + extraTrainingData,
            batch_sampler=trainBatchSampler,
            **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0])
            + extraTestData,
            batch_sampler=testBatchSampler,
            **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0])
            + extraTrainingData,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
            **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0])
            + extraTestData,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
            **kwargs,
        )

    return train_loader, test_loader


def getCifar10(
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (32, 32)
    if size is None or _sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    trainDataset = (
        datasets.CIFAR10(
            _getRoot(dataDir),
            train=True,
            download=True,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTrainingData
    )
    testDataset = (
        datasets.CIFAR10(
            _getRoot(dataDir),
            train=False,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTestData
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(trainDataset.targets, 10, batchSize)
        testBatchSampler = BalancedBatchSampler(testDataset.targets, 10, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_sampler=trainBatchSampler, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_sampler=testBatchSampler, **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )
    return train_loader, test_loader


def getSVHN(
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> tuple:
    originalSize = (32, 32)
    if size is None or _sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    trainDataset = (
        datasets.SVHN(
            _getRoot(dataDir),
            split="train",
            download=True,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTrainingData
    )
    testDataset = (
        datasets.SVHN(
            _getRoot(dataDir),
            split="test",
            download=True,
            transform=tfms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(y)),
        )
        + extraTestData
    )

    if batchBalancing:
        trainBatchSampler = BalancedBatchSampler(trainDataset.labels, 10, batchSize)
        testBatchSampler = BalancedBatchSampler(testDataset.labels, 10, batchSize)
    else:
        trainBatchSampler = None
        testBatchSampler = None

    if batchBalancing:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_sampler=trainBatchSampler, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_sampler=testBatchSampler, **kwargs,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )

        test_loader = torch.utils.data.DataLoader(
            testDataset, batch_size=batchSize, shuffle=True, drop_last=True, **kwargs,
        )
    return train_loader, test_loader


def getDatasets(
    names: Union[str, list],
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
    batchBalancing: bool = False,
    extraTrainingData: TensorDataset = [],
    extraTestData: TensorDataset = [],
    dataDir: str = None,
    **kwargs,
) -> Union[tuple, list]:
    datasetMap = {
        "mnist": lambda x: getMNIST(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
        "MNIST": lambda x: getMNIST(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
        "notMnist": lambda x: getNotMNIST(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
        "notMNIST": lambda x: getNotMNIST(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
        "cifar10": lambda x: getCifar10(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
        "cifar100": lambda x: getCifar100(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
        "fashion": lambda x: getFashion(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
        "svhn": lambda x: getSVHN(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
        "traffic": lambda x: getTrafficSigns(
            x,
            size,
            channels,
            batchBalancing,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
            dataDir=dataDir,
            **kwargs,
        ),
    }

    datasetMap.update(
        {
            f"mnist_{i}{i+1}": lambda x, i=i: getPairwiseMNIST(
                x,
                (i, i + 1),
                size=size,
                channels=channels,
                batchBalancing=batchBalancing,
                extraTrainingData=extraTrainingData,
                extraTestData=extraTestData,
                dataDir=dataDir,
                **kwargs,
            )
            for i in range(9)
        }
    )

    datasetMap.update(
        {
            f"cifar100_{i}": lambda x, i=i: getIncrementalCifar100(
                x,
                i,
                size=size,
                channels=channels,
                batchBalancing=batchBalancing,
                extraTrainingData=extraTrainingData,
                extraTestData=extraTestData,
                dataDir=dataDir,
                **kwargs,
            )
            for i in range(10)
        }
    )

    datasetMap.update(
        {
            f"TIN_{i}": lambda x, i=i: getTinyImageNet(
                x,
                i,
                size=size,
                channels=channels,
                batchBalancing=batchBalancing,
                extraTrainingData=extraTrainingData,
                extraTestData=extraTestData,
                dataDir=dataDir,
                **kwargs,
            )
            for i in range(10)
        }
    )

    if type(names) is str or len(names) == 1:
        if type(names) is str:
            loaders = datasetMap[names](batchSize)
        else:
            loaders = datasetMap[names[0]](batchSize)
    else:
        loaders = [datasetMap[name](batchSize) for name in names]
    return loaders
