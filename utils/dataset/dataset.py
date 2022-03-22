import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import PIL
import os
import pandas as pd

class MHCoverDataset(Dataset):
    """
    Class defines custom Dataset as a workaround to the ImageFolder class
    """
    def __init__(self, root_dir: str, df: pd.DataFrame, transform: 'Compose' = None,
                 label_indexer: str = 'concatenated_type'):
        """
        Initializes the dataset class.

        Params:
        ------------------
        root_dir: str
            Defines the path from where all images should be imported from

        df: pd.DataFrame
            Prefiltered Dataframe to load labels from (filtered according to Train-Val-test set)

        transform: torch.utils.transforms.Compose
            Compose of different transforms applied during import of an image.

        label_indexer: str
            Column name of the index to take.

        _images: list
            List of all images names in the root dir.
        """
        self.root_dir = root_dir
        self.df = df
        self.transform = transform
        self.label_indexer = label_indexer
        self._images = self._init_images()

    def _init_images(self):
        """Matches Data in Folder with the ones in the filtered pd.DataFrame"""
        all_images = [img for img in os.listdir(self.root_dir) if '.jpg' in img]
        self.df = self.df[self.df['image'].isin(all_images)]

        return self.df['image'].to_list()

    def __len__(self):
        """Returns length of Dataset"""
        return len(self._images)

    def __getitem__(self, idx):
        """Returns Item with given IDX"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_name = os.path.join(self.root_dir, self._images[idx])
        image = PIL.Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        # Load label
        label = self.df.loc[idx, self.label_indexer]

        return image, label


def get_dataloader(root_dir: str, df: pd.DataFrame, transforms: 'torch.utils.Compose',
                   batch_size: int, workers: int):
    """
    Function returns a dataloader with given parameters

    Params:
    ---------------
    root_dir: str
        Defines the path from where all images should be imported from

    df: pd.DataFrame
            Prefiltered Dataframe to load labels from (filtered according to Train-Val-test set)

    transform: torch.utils.transforms.Compose
        Compose of different transforms applied during import of an image.

    batch_size; int
        Size of the imported batch

    workers: int
        Amount of CPU workers for the loading of data into gpu.
    """

    custom_dataset = MHCoverDataset(root_dir=root_dir, df=df, transform=transforms)

    return DataLoader(dataset=custom_dataset,
                      batch_size=batch_size,
                      num_workers=workers)
