from typing import Tuple
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from cl_datasets.loader import getDatasets, supported_datasets


class Cl_dataset:
    def __init__(
        self,
        name: str,
        nClasses: int,
        nTasks: int = 1,
        trainTransforms: torch.nn.Module = None,
        testTransforms: torch.nn.Module = None,
        previousDataset=None,
    ) -> None:
        """Supervised continual learning dataset wrapper

        Args:
            name (str): Name (must be in the list of supported datasets)
            nClasses (int): Total number of classes for this dataset
            nTasks (int, optional): Total number of tasks for this dataset. Defaults to 1.
            trainTransforms (torch.nn.Module, optional): Training transformations. Defaults to None.
            testTransforms (torch.nn.Module, optional): Test transformations. Defaults to None.
            previousDataset (Cl_dataset, optional): Previous dataset in a series of continual training datasets. Defaults to None.
        """
        self.name = name  # format must respect known tasks in cl_datasets
        try:
            assert self.name in supported_datasets
        except AssertionError:
            raise (f"{name} is not in supported list of datasets")
        self.previous = previousDataset
        self.nClasses = nClasses
        self.nTasks = nTasks
        self.trainTransforms = trainTransforms
        self.testTransforms = testTransforms

    @property
    def lastTaskIdx(self):
        return self.nTasks - 1

    @property
    def nClassesPerTask(self):
        return self.nClasses // self.nTasks

    @property
    def isIncremental(self):
        return self.nTasks > 1

    def getTaskDataloaders(
        self, taskIdx: int, batchSize: int, replayBufferSize: int = 0
    ) -> Tuple[TensorDataset, TensorDataset]:
        if taskIdx == 0:
            train_loader, test_loader = getDatasets(
                f"{self.name}_{taskIdx}",
                batchSize,
                trainTransforms=self.trainTransforms,
                testTransforms=self.testTransforms,
            )
        else:
            nElementsPerTask = replayBufferSize // taskIdx
            extraTrainingData, extraTestData = self.getPreviousTaskData(
                taskIdx, nElementsPerTask
            )
            train_loader, test_loader = getDatasets(
                f"{self.name}_{taskIdx}",
                batchSize,
                trainTransforms=self.trainTransforms,
                testTransforms=self.testTransforms,
                extraTrainingData=extraTrainingData,
                extraTestData=extraTestData,
            )

        return train_loader, test_loader

    def getPreviousTaskData(
        self, currentTaskIdx: int, maxSamplesPerTask: int
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Gets data from previous task using a fixed number of
        maximum samples per task for training.

        Args:
            currentTaskIdx (int): Current task index
            maxSamplesPerTask (int): Maximum samples per task (training only)

        Returns:
            Tuple[TensorDataset, TensorDataset]: Extra train data,
                                                 extra test data
        """
        trainDataBuffer = []
        testDataBuffer = []
        for previousTask in range(currentTaskIdx):
            previousTrainLoader, previousTestLoader = getDatasets(
                f"{self.name}_{previousTask}",
                1,
                trainTransforms=self.trainTransforms,
                testTransforms=self.testTransforms,
            )
            trainDataBuffer.extend(
                [(x.squeeze(0), y.squeeze()) for x, y in previousTrainLoader][
                    :maxSamplesPerTask
                ]
            )
            testDataBuffer.extend(
                [(x.squeeze(0), y.squeeze()) for x, y in previousTestLoader]
            )

        if not self.previous is None:
            # recursive call to get all data from previous datasets
            (
                previousTrainDataset,
                previousTestDataset,
            ) = self.previous.getPreviousTaskData(
                self.previous.lastTaskIdx, maxSamplesPerTask
            )
            trainDataBuffer.extend(
                [(x.squeeze(0), y.squeeze()) for x, y in previousTrainDataset]
            )
            testDataBuffer.extend(
                [(x.squeeze(0), y.squeeze()) for x, y in previousTestDataset]
            )

        xTensor = torch.stack([x for x, _ in trainDataBuffer])
        yTensor = torch.stack([y for _, y in trainDataBuffer])
        extraTrainingData = TensorDataset(xTensor, yTensor)

        xTensor = torch.stack([x for x, _ in testDataBuffer])
        yTensor = torch.stack([y for _, y in testDataBuffer])
        extraTestData = TensorDataset(xTensor, yTensor)
        return extraTrainingData, extraTestData
