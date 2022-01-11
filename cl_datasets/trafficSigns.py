"""
Traffic signs dataset
"""

import os
import numpy as np
import pickle
import urllib.request
from torch.utils.data import Dataset
from PIL import Image


class TrafficSigns(Dataset):
    """`German Traffic Signs <http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.

    """

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "traffic_signs_dataset.zip"
        self.url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip"
        # Other options for the same 32x32 pickled dataset
        # url="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip"
        # url_train="https://drive.google.com/open?id=0B5WIzrIVeL0WR1dsTC1FdWEtWFE"
        # url_test="https://drive.google.com/open?id=0B5WIzrIVeL0WLTlPNlR2RG95S3c"

        fpath = os.path.join(root, self.filename)
        training_file = os.path.join(root, "lab 2 data/train.p")
        testing_file = os.path.join(root, "lab 2 data/test.p")

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
            with open(training_file, mode="rb") as f:
                train = pickle.load(f)
            self.data = train["features"]
            self.labels = train["labels"]
        else:
            with open(testing_file, mode="rb") as f:
                test = pickle.load(f)
            self.data = test["features"]
            self.labels = test["labels"]

        self.data = np.transpose(self.data, (0, 3, 1, 2))

    @property
    def nClasses(self) -> int:
        return 43

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

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

    print("Testing Traffic signs loader")
    normalize = transforms.Normalize(
        (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669),
    )
    # normalization is not great for visualization
    # tfms = transforms.Compose([transforms.ToTensor(), normalize])
    tfms = transforms.Compose([transforms.ToTensor()])
    trainLoader = DataLoader(
        TrafficSigns("_data", transform=tfms, download=True),
        batch_size=1,
        shuffle=True,
    )
    for idx, data in enumerate(trainLoader):
        img, lbl = data
        plt.title(f"Class: {lbl.squeeze().item()}")
        plt.imshow(img.squeeze().permute(1, 2, 0))
        plt.show()

        if idx > 10:
            break
