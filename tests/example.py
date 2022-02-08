from cl_datasets import getDatasets, labelStats


def run():
    datasets = ["svhn", "cifar10", "fashion", "mnist"]
    batchSize = 32
    dataSize = (32, 32)
    nChannels = 3

    dataloaders = getDatasets(datasets, batchSize, dataSize, nChannels)

    for taskName, taskData in zip(datasets, dataloaders):
        print(f"Task: {taskName}")
        trainLoader, testLoader = taskData
        print(f"Training data samples: {len(trainLoader.dataset)}")
        print(f"Test data samples: {len(testLoader.dataset)}")
        print(f"Train data label histogram:\n{labelStats(trainLoader)}")
        print(f"Test data label histogram:\n{labelStats(testLoader)}")
    return


if __name__ == "__main__":
    run()
