import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import PIL
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import wandb
from torch.utils.data import WeightedRandomSampler
from torchvision import models
from torchvision import transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from tqdm import tqdm
from dataset import get_dataloader
from train_interface import TrainingInterface


def init_transforms(ra_num_ops: int, ra_magnitude: int):
    """Initializes Transforms"""
    train_transformations = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transformations = transforms.Compose(

        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    inverse_transforms = transforms.Compose(
        [
            transforms.Normalize(mean=[0, 0, 0], 
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], 
                                 std=[1, 1, 1]),
        ]
    )
    return (train_transformations, test_transformations, inverse_transforms)

    

def load_and_get_weights_train(df): 
    """Loads and get the weights of the train dataframe"""
    train = df[df["set"] == "train"].sample(frac=1).reset_index(drop=True)
    
    # Get the inverse label weight
    label_counts = train.shape[0] / (
    train["label"].unique().shape[0] * train["label"].value_counts()
    )
    train["weight"] = train["label"].apply(lambda x: label_counts[x])
    
    return train

    
def evaluate_max_score(model, epoch_space, dataloader_train, dataloader_test):
    """Runs through epochs and calculates test metrics"""
    scores = {"test": [], "train": []}
    for ep in tqdm(epoch_space):
        model.load_from_history(epoch=ep, inplace=True)
        metrics = model.calculate_metrics(
            dataloader_train=dataloader_train,
            dataloader_test=dataloader_test,
            metric_funcs=[precision_score, recall_score, f1_score],
            disable_pbar=True,
            average="macro",
            zero_division=0,
        )
        scores["train"].append(metrics["train"]["f1_score"])
        scores["test"].append(metrics["test"]["f1_score"])

    return scores
 
def get_max_score_from_epoch(scores, epoch_space):
    """Returns max score from epoch."""
    tmp = pd.DataFrame.from_dict(scores)
   
    return epoch_space[np.argmax(tmp["test"])]
    
    
def main(wandb):
    
    print(20*'-', 'Load History', 20*'-')

    if 'sample_history.pkl' in os.listdir():
        with open('sample_history.pkl', 'rb') as pkl_file:
            sample_history = pickle.load(pkl_file)
            sample_history.append(None)
    else:
        sample_history = [None]
        
    print(20*'-', 'Load Dataframe', 20*'-')
    df = pd.read_csv("../../data/labels.csv") 
    train = load_and_get_weights_train(df)

    # Train iterative
    ####### Hyperparameters ########
    NAME = 'resnet18_wandb'

    print(20*'-', 'Sample Parameters', 20*'-')
    hyper_space = None
    while hyper_space in sample_history:
        ## Sampler
        WEIGHTED_SAMPLER = np.random.choice([True, False])
        SAMPLER_NUM_SAMPLES = np.random.randint(1000, 3000)
        SAMPLER_REPLACEMENT = True

        ## General
        WORKERS = 20
        BATCH_SIZE = int(np.random.choice([16, 32, 64, 128]))

        ## Model
        PRETRAINED = True

        ## Training
        LR = np.random.uniform(.000001, .0001)
        WEIGHT_DECAY = np.random.uniform(.2, .6)
        WEIGHTED_LOSS = np.random.choice([True, False])
        #EPOCHS = 2
        EPOCHS = 120

        # RandAugment Param
        NUM_OPS = np.random.randint(1, 4)
        MAGNITUDE = np.random.randint(4, 10)

        hyper_space = [WEIGHTED_SAMPLER, SAMPLER_NUM_SAMPLES, PRETRAINED, 
                       LR, WEIGHT_DECAY, WEIGHTED_LOSS, EPOCHS]
    sample_history.append(hyper_space)

    # init Transforms
    print(20*'-', 'Get Dataloaders', 20*'-')
    train_transformations, test_transformations, inverse_transforms = init_transforms(ra_num_ops=NUM_OPS, ra_magnitude=MAGNITUDE)

    # Init wandbiases
    configs_wandb = {
        'MODEL_NAME': NAME, 
        'WEIGHTED_SAMPLER': WEIGHTED_SAMPLER,
        'SAMPLER_NUM_SAMPLES': SAMPLER_NUM_SAMPLES,
        'SAMPLER_REPLACEMENT': SAMPLER_REPLACEMENT,
        'WORKERS': WORKERS,
        'BATCH_SIZE': BATCH_SIZE,
        'PRETRAINED': PRETRAINED,
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WEIGHTED_LOSS': WEIGHTED_LOSS,
        'EPOCHS' : EPOCHS,
        'RA_NUM_OPS': NUM_OPS,
        'RA_MAGNITUDE': MAGNITUDE
    }
    print(configs_wandb)
    run.config.update(configs_wandb, allow_val_change=True)
    
    print(20*'-', 'Init Samplers', 20*'-')
    ########### Sampler #############
    if WEIGHTED_SAMPLER:
        sampler = WeightedRandomSampler(
            weights=train["weight"].to_numpy(),
            num_samples=SAMPLER_NUM_SAMPLES,
            replacement=SAMPLER_REPLACEMENT,
        )
        
    if WEIGHTED_SAMPLER:
        dataloader_train = get_dataloader(
            root_dir="../../data/train/",
            df=train,
            fp_label_translator="../dataset/label_translate.pkl",
            transformations=train_transformations,
            batch_size=BATCH_SIZE,
            workers=WORKERS,
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
        )
    else:
        dataloader_train = get_dataloader(
            root_dir="../../data/train/",
            df=train,
            fp_label_translator="../dataset/label_translate.pkl",
            transformations=train_transformations,
            batch_size=BATCH_SIZE,
            workers=WORKERS,
            pin_memory=True,
            shuffle=True,
        )
        
    dataloader_val = get_dataloader(
        root_dir="../../data/val/",
        df=df[df["set"] == "val"].reset_index(drop=True),
        fp_label_translator="../dataset/label_translate.pkl",
        transformations=test_transformations,
        batch_size=BATCH_SIZE,
        workers=12,
        pin_memory=True,
        shuffle=True,
    )
    dataloader_test = get_dataloader(
        root_dir="../../data/test/",
        df=df[df["set"] == "test"].reset_index(drop=True),
        fp_label_translator="../dataset/label_translate.pkl",
        transformations=test_transformations,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        pin_memory=True,
        shuffle=True,
    )

    # Test correctness of labels 
    assert dataloader_train.dataset.label_dict == dataloader_val.dataset.label_dict
    assert dataloader_val.dataset.label_dict == dataloader_test.dataset.label_dict

    # Load model
    print(20*'-', 'Prepare Model', 20*'-')
    resnet_ = models.resnet18(pretrained=PRETRAINED, progress=True)
        
    if PRETRAINED:
        for param in resnet_.parameters():
            param.requires_grad = False

        # Replace fc
        resnet_.fc = nn.Linear(512, len(dataloader_train.dataset.label_dict.keys()))

        # Enable grad
        resnet_.fc.weight.requires_grad = True
        resnet_.fc.bias.requires_grad = True

        # check
        for name, param in resnet_.named_parameters():
            if param.requires_grad:
                print("Requires Grad:", name)
    else:
        resnet_.fc = nn.Linear(512, len(dataloader_train.dataset.label_dict.keys()))

    resnet = TrainingInterface(
        model=resnet_, 
        name=NAME,
        history=True, 
        writer=wandb
    )

    if WEIGHTED_LOSS:
        train.groupby("label").first()["weight"]
        ce_weights = [
            train.loc[train["label"] == key, "weight"].iloc[0]
            for key in dataloader_train.dataset.label_dict.keys()
        ]
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(ce_weights))
        optimizer = optim.Adam(
            resnet.model.parameters(), 
            lr=LR, 
            weight_decay=WEIGHT_DECAY
        )
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            resnet.model.parameters(), 
            lr=LR, 
            weight_decay=WEIGHT_DECAY
        )
    print(20*'-', 'Start Training', 20*'-')
    ## Start Trainnig
    resnet.train(
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=EPOCHS,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        verbose=True,
        score_func=f1_score,
        average='macro'
    )

    print(20*'-', 'Start Backrun', 20*'-')
    ## Backrun epochs for calc the metrics
    #epoch_space = np.linspace(
    #    start=10, 
    #    stop=EPOCHS, 
    #    num=EPOCHS//2, 
    #    dtype=int
    #)
    #scores = evaluate_max_score(
    #    resnet, 
    #    epoch_space,
    #    dataloader_train, 
    #    dataloader_test
    #)

    # Log best results
    #run.log({'Max-F1': np.max(scores['test'])})
    #run.log({'Max-F1-EP': epoch_space[np.argmax(scores['test'])]})

    with open('sample_history.pkl', 'wb') as pkl_file:
        pickle.dump(sample_history, pkl_file)            

            
if __name__ == '__main__':
    
    iteration = 0
    while True:
        # init wandb run
        run = wandb.init(project="manhole_cover_classification", entity="sifi")
        main(run)
        run.finish()
        iteration += 1
        