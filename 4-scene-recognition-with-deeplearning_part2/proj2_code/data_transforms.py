'''
Contains functions with different data transforms
'''

from typing import Tuple

import numpy as np
import torchvision.transforms as transforms
import PIL


def get_fundamental_transforms(inp_size: Tuple[int, int],
                               pixel_mean: np.array,
                               pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the core transforms needed to feed the images to our model

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean of the raw dataset [Shape=(1,)]
  - pixel_std: the standard deviation of the raw dataset [Shape=(1,)]
  Returns:
  - fundamental_transforms: transforms.Compose with the fundamental transforms
  '''

  return transforms.Compose([transforms.Resize(inp_size),
                             transforms.ToTensor(), 
                             transforms.Normalize(mean=pixel_mean,
                                                 std = pixel_std)
      ############################################################################
      # Student code begin
      ############################################################################

      ############################################################################
      # Student code end
      ############################################################################
  ])


def get_data_augmentation_transforms(inp_size: Tuple[int, int],
                                     pixel_mean: np.array,
                                     pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the data augmentation + core transforms needed to be applied on the train set. Put data augmentation transforms before code transforms. 

  Note: You can use transforms directly from torchvision.transforms

  Suggestions: Jittering, Flipping, Cropping, Rotating.

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean of the raw dataset
  - pixel_std: the standard deviation of the raw dataset
  Returns:
  - aug_transforms: transforms.compose with all the transforms
  '''

  return transforms.Compose([transforms.Resize(inp_size),
                             transforms.ColorJitter(brightness=0.08 ,hue=0.08, saturation=0.08),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                             transforms.ToTensor(), 
                             transforms.Normalize(mean=pixel_mean,
                                                 std = pixel_std)])
