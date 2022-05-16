import unittest
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torchvision import transforms as transforms
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_pil_image


from plots import EvaluationPlots
from unittest.mock import patch
import pickle
#import sys
#sys.path.append('../../')
#from utils import get_dataloader


pkl_url_file = "../dataset/label_translate.pkl"
#root_dir = "../data/images_transformed/"
#df = pd.read_csv("../../data/labels.csv")

with open(pkl_url_file, 'rb') as pkl_file:
    label_dict = pickle.load(pkl_file)

#model = torch.load('../../model/model_resnet152.pth')
#my_transforms = transforms.Compose(
#        [
#            transforms.Resize(224),
#            transforms.ToTensor(),
#        ]
#    )
#testloader = get_dataloader(
#    root_dir=root_dir,
#    df=df[df.set =="test"],
#    fp_label_translator=pkl_url_file,
#    transformations=my_transforms,
#    batch_size=32,
#    workers=0,
#    pin_memory=True,
#    shuffle=False
#)



class TestEvaluationPlots(unittest.TestCase):
    #test if plot_train_val_loss works with train_loss and/or val_loss as list or array
    @patch("plots.plt.show")
    def test_plot_train_val_loss(self, mock_show):
        train_loss = [1,2,3,4,5]
        val_loss = [1,2,3,4,5]
        EvaluationPlots.plot_train_val_loss(train_loss, val_loss)
        EvaluationPlots.plot_train_val_loss(train_loss)
        EvaluationPlots.plot_train_val_loss(val_loss)

    @patch("plots.plt.show")
    def test_plot_confusion_matrix(self,mock_show):
        y_true = torch.tensor([1,2,3,4,5])
        y_pred = torch.tensor([1,2,3,4,5])
        EvaluationPlots.plot_confusion_matrix(y_true, y_pred, label_dict)

    @patch("plots.plt.show")
    def test_loss_sma(self, mock_show):
        train_loss = [1,2,3,4,5]
        val_loss = [1,2,3,4,5]
        EvaluationPlots.loss_sma(train_loss)
        EvaluationPlots.loss_sma(val_loss)

    #@patch("plots.plt.show")
    #def test_visualize_model(self, mock_show):
    #this part is not used for new files anymore and will be removed in the future
  

    @patch("plots.plt.show")
    def test_plot_pred(self, mock_show):
        y_true = torch.tensor([1])
        y_pred = torch.tensor([1]) 
        y_prob = torch.tensor([[1,2,3]])
        y_images = torch.tensor([[[.1]],[[.4]],[[.3]],[[1]]])
        EvaluationPlots.plot_pred(y_true, y_pred, y_prob, y_images,1, incorrect = False,label_dict=label_dict)
        

    @patch("plots.plt.show")
    def test_plot_metrics(self, mock_show):
        metrics ={'train': {'precision_score': 0.8810113130762057,
                    'recall_score': 0.8773985005466114,
                    'f1_score': 0.8778944347986286},
          'test': {'precision_score': 0.621665267766038,
                   'recall_score': 0.6030086019666873,
                   'f1_score': 0.5322444162155169}}
        EvaluationPlots.plot_metrics(metrics)


    @patch("plots.plt.show")
    def test_metric_evaluation(self, mock_show):
        y_true = torch.tensor([1,2,3,4,5])
        y_pred = torch.tensor([1,2,3,4,5])
        EvaluationPlots.metric_evaluation(y_true, y_pred, label_dict)
