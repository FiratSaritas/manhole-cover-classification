"""
This file is for moving all images within given folder into three different subfolder
train-val-test.
"""
import pandas as pd
import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool

DF_PATH = '../../data/transformation_split.csv'

FROM_FOLDER = '../../data/images_transformed/'
TRAIN_FOLDER = '../../data/train1/'
VAL_FOLDER = '../../data/val1/'
TEST_FOLDER = '../../data/test1/'


if __name__ == '__main__':
    
    # Load Df
    df = pd.read_csv(DF_PATH)
    df = df.drop_duplicates()
    
    # Load image names
    train_images = set(df.loc[df['set'] == 'train', 'image'].to_list())
    val_images = set(df.loc[df['set'] == 'val', 'image'].to_list())
    test_images = set(df.loc[df['set'] == 'test', 'image'].to_list())
    
    # Create folder
    for fname in [TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER]:
        if os.path.isdir(fname) == False:
            if input(f'Create Folder {fname}? y/n ') == 'y':
                os.mkdir(fname)
    
    for all_images, new_folder in [(train_images, TRAIN_FOLDER), 
                                   (val_images, VAL_FOLDER),
                                   (test_images, TEST_FOLDER)]:
        for img_name in tqdm(all_images):
            fp_from = os.path.join(FROM_FOLDER, img_name)
            fp_to = os.path.join(new_folder, img_name)
            try:
                shutil.move(fp_from, fp_to)
            except FileNotFoundError as fe:
                print(fe, img_name)