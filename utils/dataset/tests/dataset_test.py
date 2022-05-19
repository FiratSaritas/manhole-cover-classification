import unittest
import torch
from torchvision import transforms
import pandas as pd
import sys
sys.path.append('../')
from dataset import get_dataloader


class TestMHCoverDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        #load data
        my_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        )


        pkl_url_file = "../label_translate.pkl"
        root_dir = "../../../data/images_transformed/"
        df = pd.read_csv("../../../data/labels.csv")


        seg_dataloader = get_dataloader(root_dir=root_dir,
                                        df=df[df.set =="train"],
                                        fp_label_translator=pkl_url_file,
                                        transformations=my_transforms,
                                        batch_size=32,
                                        workers=0,
                                        pin_memory=True,
                                        shuffle=True)

        cls.samples = next(iter(seg_dataloader))

    def test_image_tensor_dimensions(self):
        #test if image tensor has correct dimensions
        image_tensor_shape = TestMHCoverDataset.samples[0].shape
        self.assertEqual(image_tensor_shape[0], 32)
        self.assertEqual(image_tensor_shape[1], 3)
        self.assertEqual(image_tensor_shape[2], 224)
        self.assertEqual(image_tensor_shape[3], 224)

    def test_label_type(self):
        #test if image is a tensor
        self.assertEqual(type(TestMHCoverDataset.samples[0]), torch.Tensor)


    def test_batch_size(self):
        #test if batch size is 32
        self.assertEqual(len(TestMHCoverDataset.samples[0]), 32)

    def test_label_range(self):
        #test if labels all between 0 and 12
        for label in TestMHCoverDataset.samples[1]:
            self.assertTrue(label >= 0)
            self.assertTrue(label <= 12)