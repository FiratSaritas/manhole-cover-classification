# manhole-cover-classification

![ezgif-4-b657bbd0b1](https://user-images.githubusercontent.com/82641568/169886162-626b7faf-20ef-469b-ae19-8d1b06e74683.gif)
![ezgif-5-35c28f41eb](https://user-images.githubusercontent.com/82641568/169886048-32407b46-122a-4614-bbc0-123ecd151809.gif)
![ezgif-4-1abd2a1947](https://user-images.githubusercontent.com/82641568/169886404-4e68385d-f0f9-4674-871d-7cc4c3b0c60b.gif)


## About project
The aim of this project is to develop an image classification model, which should be able to classify different types of manhole covers . We don't have enough images to train the model. Sometimes we also have pictures of water pipe covers instead of manhole covers.
The available images are all unlabelled. Now we have to find a solution to ensure correct classification of manhole covers.

## Folder Structure Conventions

```
    ├── Checklists             # Checklists for a clean code and project (files type: pdf)
    ├── augmentation           # Use of different data augmentations (files type: ipynb)
    ├── data                   # Data of the project (files type: csv)
    │   ├── archive
    ├── eda                    # Exploratory data analysis (files type: ipynb)
    ├── model                  # Different trained models (files type: ipynb)
    │   ├── final              # Calls final model and makes a prediction (files type: py)
    ├── utils                  # outsourced functions (files type: py)
    │   ├── archive
    │   ├── dataset            # dataset functions (files type: py)
    │   │   ├── tests
    │   ├── plots              # plot functions (files type: py)
    │   │   ├── tests
    │   ├── training           # training functions (files type: py)
    │   │   ├── tests
    └── README.md             
    └── git_workflow.md
    └── requirements.txt
```

## Getting Started

### Installation
Clone project locally
 ```sh
    git@github.com:FiratSaritas/manhole-cover-classification.git
 ```

### Downloads

#### Model:
Download model from Google Drive and add it to the folder ./model here:
**link** (not ready yet)

#### Images (optional):
Download images as Folder (images_transformed) from Google Drive and add it to the folder ./data here:
[**link** (not ready yet)](https://drive.google.com/drive/folders/1y5T1-WUZB1Vsp87aBiU6hDxagiY2mGgi?usp=sharing)



### Prerequisites 
Install required packages
 ```sh
    pip install requirements.txt
 ```

### Run project

### Run test

There are tests for the outsourced python files. These python files are located in the "utils" folder. There are subfolders of the corresponding classes (e.g. dataset). There is a folder with the name "tests" and there are the unittests. 
You can run a test with the following code:
```sh
    python -m unittest [name of the testfile]
 ```

## Project status
This project is still in progress.

## Contact

firat.saritas@students.fhnw.ch<br />
simon.staehli@students.fhnw.ch<br />
kajenthini.kobivasan@students.fhnw.ch
