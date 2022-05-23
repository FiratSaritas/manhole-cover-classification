import unittest
import torch
import re
import pickle
import pandas as pd
import PIL
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as transforms
from sklearn.metrics import precision_score,recall_score,f1_score
import sys
sys.path.append('../../../')
from utils import MHCoverDataset, get_dataloader
from utils.training import TrainingInterface


model = torch.load('model_test.pth')

my_transforms = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
    ])

pkl_url_file = "../../../utils/dataset/label_translate.pkl"
root_dir = "../../../data/images_transformed/"
df = pd.read_csv("../../../data/labels.csv")

with open(pkl_url_file, 'rb') as pkl_file:
    label_dict = pickle.load(pkl_file)

seg_dataloader = get_dataloader(root_dir=root_dir,
                                df=df[df.set =="train"],
                                fp_label_translator=pkl_url_file,
                                transformations=my_transforms,
                                batch_size=32,
                                workers=0,
                                pin_memory=True,
                                shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0001)

class TestTrainingInterface(unittest.TestCase):
    def test_print_network(self):
        mm = model.model
        mp = model.print_network()
        same = mm == mp
        self.assertEqual(same, True)


    def test_print_total_params(self):
        mp = model.print_total_params(return_=True)
        parameter = re.findall(r'\d+', mp)  
        self.assertEqual(int(parameter[0]), 11689512)


    def test_train(self):
        train = model.train(
            criterion=criterion,
            optimizer=optimizer,
            n_epochs=1,
            dataloader_train=seg_dataloader,
            dataloader_val=seg_dataloader,
            verbose=True,)

        ptp = train.print_total_params(return_=True) == model.print_total_params(return_=True)
        self.assertEqual(ptp, True)



    def test_predict(self):
        predict = model.predict(dataloader=seg_dataloader)
        self.assertEqual(type(predict[0]), torch.Tensor)
        self.assertEqual(type(predict[1]), torch.Tensor)
        self.assertEqual(type(predict[2]), torch.Tensor)



    def test_predict_one(self):
        image = PIL.Image.open("image_test.png")
        image = my_transforms(image)
        image = image.unsqueeze(0)
        pred_type = type(model.predict_one(image,label_dict))
        self.assertEqual(pred_type, dict)

    def test_metrics(self):
        metrics = model.calculate_metrics(
            dataloader_train=seg_dataloader,
            dataloader_test=seg_dataloader,
            metric_funcs=[precision_score, recall_score, f1_score],
            average="macro",)
        self.assertEqual(type(metrics), dict)
                

