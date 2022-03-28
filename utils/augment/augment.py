"""
This script is for offline Augmentation purposes. It depends on the parameters withing augment_config.yaml

When calling this File:
- It will create a new folder given as parameter TO_FOLDER
- Make a transformed copy of all images from 'FROM_FOLDEr' to 'TO_FOLDER'
- Create a new dataframe with the new filenames which is adaption of the given dataframe 'DF_PATH_FROM'

"""
import os
import time
import yaml
import PIL.Image
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

import torch
from torchvision import transforms as transforms


class RandomAugmentor(object):
    """
    This class implements random transformation functions from torchvision.
    """
    image_size = 128
    augmentations = {
        'orig': transforms.Resize(image_size), # Blind transformation
        'gray': transforms.Grayscale(num_output_channels=3),
        'jit': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        'fliph': transforms.RandomHorizontalFlip(p=1),
        'flipv': transforms.RandomVerticalFlip(p=1),
        'pers': transforms.RandomPerspective(distortion_scale=0.5, p=1, fill=0),
        'rot': transforms.RandomRotation(degrees=360),
        'blur': transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    }
    
    def __init__(self, apply_n: int, reuse_transform: bool = False, replace_sample: bool = False):
        """
        Random Chosen transformation for a given image.
        
        Params:
        -------------------
        apply_n: int
            How many transformation in sequence order to apply
        
        replace_sample: int
            random choice param if to sample from same list again.
            
        reuse_transform: bool
            if to reuse the transfrom for subsequent transformations
        
        current_transforms:
            Currently applied transformations-
        """
        self.apply_n = apply_n
        self.replace_sample = replace_sample
        self.reuse_transform = reuse_transform
        self.current_transforms = None
    
    def _get_transformation(self):
        """Returns random Transformations"""
        if not self.reuse_transform or self.current_transforms is None:
            self.current_transforms = np.random.choice(list(RandomAugmentor.augmentations.keys()), 
                                                       size=self.apply_n, 
                                                       replace=self.replace_sample)
        transforms_ = [RandomAugmentor.augmentations[t] for t in self.current_transforms]
        
        return transforms.Compose(transforms_)
    
    def _get_transformation_given_k(self, given_k: list):
        """Returns transformations given K transformation steps."""
        transforms_ = [RandomAugmentor.augmentations[t] for t in given_k]
        
        return transforms.Compose(transforms_)

    
    def random_transform(self, image: PIL.Image):
        """Calls Random Transformations to transforn an image."""
        transform = self._get_transformation()
        image = transform(image)
        
        return image
    
    def get_keys(self):
        """Returns transformation keys"""
        return self.current_transforms
    
    def transform_given_key(self, image: PIL, given_key):
        """Transforms an image given a key."""
        if given_key:
            transform = self._get_transformation_given_k(given_k=given_key)
            image = transform(image)

        return image
           
def _transform_image(image_paths: tuple):
    """
    MP Func - Transform Single Image given with path
    
    Params:
    --------------
    image_paths: tuple
        Tuple of two path (from , to)
        example: [(../data/image1.jpg, ../data/processed/image1.jpg), ...]
    
    Returns:
    --------------
    Saves image given in the last place of the tuple 'image_paths'
    """
    # load image
    image = PIL.Image.open(image_paths[0])
    
    # apply base transformation
    image = base_transforms(image)
    
    # apply random augmentations
    image = random_augmenter.random_transform(image)
    applied_trans = random_augmenter.get_keys()
    
    # Save IMage with transformation in name
    tmp_path = image_paths[1].split('.png')[0]
    image.save(tmp_path + '_' + '_'.join(applied_trans) + '.png')
    


if __name__ == '__main__':
    # Load Configs
    with open('augment_config.yaml', 'r') as yaml_file:
        conf = yaml.load(yaml_file, yaml.FullLoader)
    print('Configurations:\n' , conf)
    if not os.path.isdir(conf['TO_FOLDER']):
        os.mkdir(conf['TO_FOLDER'])
        print('\nCreated:', conf['TO_FOLDER'])
    
    # Build Iterable
    all_filenames = os.listdir(conf['FROM_FOLDER'])
    from_ = [os.path.join(conf['FROM_FOLDER'], f) for f in all_filenames]
    to_ = [os.path.join(conf['TO_FOLDER'], f) for f in all_filenames]
    mp_iterable = list(zip(from_, to_))
    if conf['TEST_ONLY']:
        mp_iterable = mp_iterable[:5]
        
    print('\nBuilt Iterable:\n', mp_iterable[:2])
    
    user_input = input('Want Proceed? (y/n)  ')
    if user_input == 'y':
        print(20*'-', f'Start at {time.ctime()}',20*'-')
        # Initiate Transformation Classes
        base_transforms = transforms.Compose([transforms.CenterCrop(conf['IMAGE_SIZE']-conf['REDUCE_PIXEL_CROP']),
                                              transforms.Resize(conf['RESIZE'])])
        random_augmenter = RandomAugmentor(apply_n=conf['RANDOM_APPLY_N'],
                                           reuse_transform=conf['RANDOM_REUSE_TRANSFORM'],
                                           replace_sample=conf['RANDOM_REPLACE'])
        
        # Call Multiprocessing 
        pool = Pool(processes=conf['N_PROCESSES'])
        for _ in tqdm(pool.imap_unordered(_transform_image, mp_iterable), total=len(mp_iterable)):
            pass
        
        print(20*'-', f'End at {time.ctime()}', 20*'-')
        # Change Labelled Dataframe with given new name of dataframe due to transformations
        print(50*'=')
        print('Creating Copy of labelled Dataframe with new ID')
        df = pd.read_csv(conf['DF_PATH_FROM'])
        
        # Extract old id name without ending (always first element when split _)
        df['image'] = df['image'].apply(lambda x: x.split('.')[0])
        
        # Create dict for replacement
        id_to_new = {i.split('_')[0]:i for i in os.listdir(conf['TO_FOLDER'])}
        df['image'] = df['image'].replace(id_to_new)
        
        df.to_csv(conf['DF_PATH_TO'], index=False)
        
