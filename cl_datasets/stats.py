import torch
import numpy as np


def labelStats(dataloader, largestLabel: int = None) -> np.array:
    """
    Compute a labels histogram from a dataloader

    dataloader: Dataloader
    largestLabel: (int) Largest label to consider for the histogram

    return: (numpy.array) Label histogram
    """
    labels = []
    for _, targets in dataloader:
        batchTargets = torch.flatten(targets)
        labels.extend([label.item() for label in batchTargets])
    if largestLabel is None:
        maxLabel = np.max(labels)
        histo, _ = np.histogram(labels, bins=list(range(maxLabel + 1)))
    else:
        histo, _ = np.histogram(labels, bins=list(range(largestLabel + 1)))
    return histo

