"""
notMNIST dataset
"""

import os
import numpy as np
import pickle
import urllib.request
from torch.utils.data import Dataset
from PIL import Image


class NotMNIST(Dataset):
    """The notMNIST dataset is an image recognition dataset of font glypyhs 
    for the letters A through J useful with simple neural networks. It is quite 
    similar to the classic MNIST dataset of handwritten digits 0 through 9.

    Args:
        root (string): Root directory of dataset where directory ``notMNIST``
        exists.
        train (bool, optional): Indicate if training.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.

    """

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "notmnist.zip"
        self.url = "https://github.com/nkundiushuti/notmnist_convert/blob/master/notmnist.zip?raw=true"

        fpath = os.path.join(root, self.filename)
        training_file = os.path.join(root, "notmnist_train.pkl")
        testing_file = os.path.join(root, "notmnist_test.pkl")

        if os.path.isfile(training_file) and os.path.isfile(testing_file):
            datafiles_present = True
        else:
            datafiles_present = False

        if not os.path.isfile(fpath) and not datafiles_present:
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + self.url)
                self.download()

        if train:
            with open(training_file, "rb") as f:
                train = pickle.load(f)
            self.data = train["features"].astype(np.uint8)
            self.labels = train["labels"].astype(np.uint8)
        else:
            with open(testing_file, "rb") as f:
                test = pickle.load(f)

            self.data = test["features"].astype(np.uint8)
            self.labels = test["labels"].astype(np.uint8)

    @property
    def nClasses(self):
        return 10

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno

        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile

        zip_ref = zipfile.ZipFile(fpath, "r")
        zip_ref.extractall(root)
        zip_ref.close()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    print("Testing notMNIST loader")
    normalize = transforms.Normalize(0.5, 0.2724)
    # normalization is not great for visualization
    # tfms = transforms.Compose([transforms.ToTensor(), normalize])
    tfms = transforms.Compose([transforms.ToTensor()])
    trainLoader = DataLoader(
        NotMNIST("_data", transform=tfms, download=True), batch_size=1, shuffle=True,
    )
    for idx, data in enumerate(trainLoader):
        img, lbl = data
        plt.title(f"Class: {lbl.squeeze().item()}")
        plt.imshow(img.squeeze())
        plt.show()

        if idx > 10:
            break
