"""
Balanced batch generator
"""

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        # super(BalancedBatchSampler, self).__init__()
        if isinstance(labels, torch.Tensor):
            self.labels = labels.numpy()
        else:
            self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


if __name__ == "__main__":
    # load a dataet and display the contents of a batch
    from cl_datasets.loader import getDatasets

    DATASET = "mnist"
    BATCH_SIZE = 16
    train_loader, test_loader = getDatasets(DATASET, BATCH_SIZE)
    for idx, data in enumerate(train_loader):
        _, targets = data
        print(f"Labels for batch {idx}:{targets.squeeze()}")
        if idx > 9:
            break
