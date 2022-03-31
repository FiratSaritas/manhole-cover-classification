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


# Load Configs
with open('augment_config.yaml', 'r') as yaml_file:
    conf = yaml.load(yaml_file, yaml.FullLoader)
    
if conf['IS_OS_WINDOWS']:
    # Read dataframe with all transform steps 
    df_from = pd.read_csv(conf['DF_PATH']) 

    
class RandomAugmentor(object):
    """
    This class implements random transformation functions from torchvision.
    """
    
    def __init__(self, apply_n: int, reuse_transform: bool = False, replace_sample: bool = False, 
                 image_size = 128):
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
            
        image_size: int
            image size for the blind transformation
        """
        self.apply_n = apply_n
        self.replace_sample = replace_sample
        self.reuse_transform = reuse_transform
        self.current_transforms = None
        self._image_size = image_size
        self._init_transforms()
        
    def _init_transforms(self):
        RandomAugmentor.augmentations = {
            'orig': transforms.Resize(self._image_size), # Blind transformation
            'jit': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            'fliph': transforms.RandomHorizontalFlip(p=1),
            'flipv': transforms.RandomVerticalFlip(p=1),
            'pers': transforms.RandomPerspective(distortion_scale=0.5, p=1, fill=0),
            'rot': transforms.RandomRotation(degrees=360),
            'blur': transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            'raff': transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(.8, 0.9)),
            'rcrop': transforms.RandomCrop(size=(self._image_size, self._image_size)),
            'rinv': transforms.RandomInvert(),
            'rshrp': transforms.RandomAdjustSharpness(sharpness_factor=2),
            'rcont': transforms.RandomAutocontrast(p=0.5)
        }
    
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
        if given_key: # Workaround as it was filled with zeros == False
            transform = self._get_transformation_given_k(given_k=given_key)
            image = transform(image)

        return image
           
def _transform_image(mp_iterable: tuple):
    """
    MP Func - Transform Single Image given with path
    
    Params:
    --------------
    image_paths: tuple
        Tuple of two path (from , to)
        example:
            [(../data/image1.jpg, ../data/processed/image1_trans.jpg, 'orig_flipv_fliph'), ...]
            or when conf not given the transformation key with preserverd storage loc
            [(../data/image1.jpg), (../data/image2.jpg)]
    
    Returns:
    --------------
    Saves image given in the second place of the tuple 'image_paths' which indicates to new path.
    """
    # load image
    image = PIL.Image.open(mp_iterable[0])
    
    # apply base transformation
    image = pre_transforms(image)

    if len(mp_iterable) == 1:
        # Applies random augmentations
        image = random_augmenter.random_transform(image)
        trans = random_augmenter.current_transforms
        image = post_transforms(image)
        
        # Built save path
        save_path = conf['TO_FOLDER'] + mp_iterable[0].split('/')[-1].split('.')[0]
        save_path = save_path + '_' + '_'.join(trans) + '.png'
        image.save(save_path)  
    else:
        # Applies Transformations given a Transformation key
        if mp_iterable[2] != '0':
            image = random_augmenter.transform_given_key(image, mp_iterable[2].split('_'))
        image = pre_transforms(image)
        image.save(mp_iterable[1])   
        
if conf['IS_OS_WINDOWS']:
    # Initiate Transformation Classes
    pre_transforms = transforms.Compose([transforms.CenterCrop(conf['IMAGE_SIZE']-conf['REDUCE_PIXEL_CROP']),
                                         transforms.Resize(conf['RESIZE']), 
                                         transforms.Grayscale(num_output_channels=3)])
    random_augmenter = RandomAugmentor(apply_n=conf['RANDOM_APPLY_N'],
                                       reuse_transform=conf['RANDOM_REUSE_TRANSFORM'],
                                       replace_sample=conf['RANDOM_REPLACE'], 
                                       image_size=conf['RESIZE'])
    post_transforms = transforms.Compose([transforms.Resize(conf['RESIZE'])])


if __name__ == '__main__':  
    
    # Load Configs
    print('Configurations:\n' , conf)

    # Read dataframe with all transform steps 
    df_from = pd.read_csv(conf['DF_PATH'])
    
    # Create new folder
    if os.path.isdir(conf['TO_FOLDER']) == False:
        if input('DIR not exits. Create One? (y/n) ') == 'y':
            os.mkdir(conf['TO_FOLDER'])
            print('\nCreated:', conf['TO_FOLDER'])           
    
    # Build Iterable
    if conf['GIVEN_KEY']:
        tmp = list(df_from[['image', 'transforms', 'filename']].to_records(index=False))
        from_ = [os.path.join(conf['FROM_FOLDER'], fname[0]) for fname in tmp]
        transforms_ = [trans[1] for trans in tmp]
        to_ = [os.path.join(conf['TO_FOLDER'], fname[2]) for fname in tmp]
        mp_iterable = list(zip(from_, to_, transforms_))
    else:
        from_ = os.listdir(conf['FROM_FOLDER'])
        from_ = [os.path.join(conf['FROM_FOLDER'], fname) for fname in from_]
        mp_iterable = list(zip(from_))
        
    if conf['TEST_ONLY']:
        mp_iterable = mp_iterable[:20]
        
    print('\nBuilt Iterable:\n', mp_iterable[:2])
    
    user_input = input('Want Proceed? (y/n)  ')
    if user_input == 'y':
        print(20*'-', f'Start at {time.ctime()}',20*'-')
        
        # Initiate Transformation Classes
        pre_transforms = transforms.Compose([transforms.CenterCrop(conf['IMAGE_SIZE']-conf['REDUCE_PIXEL_CROP']),
                                             transforms.Resize(conf['RESIZE']), 
                                             transforms.Grayscale(num_output_channels=3)])
        random_augmenter = RandomAugmentor(apply_n=conf['RANDOM_APPLY_N'],
                                           reuse_transform=conf['RANDOM_REUSE_TRANSFORM'],
                                           replace_sample=conf['RANDOM_REPLACE'], 
                                           image_size=conf['RESIZE'])
        post_transforms = transforms.Compose([transforms.Resize(conf['RESIZE'])])
        
        
        # Call Multiprocessing 
        pool = Pool(processes=conf['N_PROCESSES'])
        for _ in tqdm(pool.imap_unordered(_transform_image, mp_iterable), total=len(mp_iterable)):
            pass
        
        if not conf['GIVEN_KEY']:
            print('Build Dataframe ...')
            # Build Dataframe
            all_images_new = os.listdir(conf['TO_FOLDER'])
            
            # Load all data from folder
            # extract image id from folder data and dataframe trans_split
            # join to dataframe
            pass
        
        print(20*'-', f'End at {time.ctime()}', 20*'-')
        print(50*'=')       
        
