"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import os
from typing import List, Tuple

import numpy as np
import PIL
import torch
import torch.utils.data as data
import torchvision

from torchvision import transforms

from proj1_code.utils import load_image, save_image

from numpy import loadtxt

def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Creates a dataset of paired images from a directory.

    The dataset should be partitioned into two sets: one contains images that
    will have the low pass filter applied, and the other contains images that
    will have the high pass filter applied.

    Args
    - path: string specifying the directory containing images
    Returns
    - images_a: list of strings specifying the paths to the images in set A,
        in lexicographically-sorted order
    - images_b: list of strings specifying the paths to the images in set B,
        in lexicographically-sorted order
    """
    
    

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    filenames =[]
    directory = os.listdir(path)
    for fname in directory:
        filenames.append(path+'/'+fname)
    images_a = []
    images_b = []
    filenames.sort()
    
    images_a = filenames[::2]
    images_b = filenames[1::2]
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return images_a, images_b


def get_cutoff_standardddeviations(path: str) -> List[int]:
    """
    Gets the cutoff standard deviations corresponding to each pair of images.

    The cutoff are the values you discovered from experimenting in
    part 2.

    Args
    - path: string specifying the path to the .txt file with cutoff standard
      deviation values
    Returns
    - List[int]. The array should have the same
      length as the number of image pairs in the dataset
    """

    cutoffs = []

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    
    cutoffs = loadtxt(path, comments="#", delimiter="\n", unpack=False)
    return cutoffs


class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        You must replace self.transform with the appropriate transform from
        torchvision.transforms that converts a PIL image to a torch Tensor. You can
        specify additional transforms (e.g. image resizing) if you want to, but
        it's not necessary for the images we provide you since each pair has the
        same dimensions.

        Args:
        - image_dir: string specifying the directory containing images
        - cf_file: string specifying the path to the .txt file with cutoff
          standard deviation values
        """
        images_a, images_b = make_dataset(image_dir)

        cutoffs = get_cutoff_standardddeviations(cf_file)

        self.transform = transforms.Compose([transforms.ToTensor()])

        ########################################################################
        # TODO: YOUR CODE HERE
        ########################################################################
        self.images_a = images_a
        self.images_b = images_b
        self.cutoffs = cutoffs
        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

    def __len__(self) -> int:
        """Returns number of pairs of images in dataset."""

        ########################################################################
        # TODO: YOUR CODE HERE
        ########################################################################
        

        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        return len(self.images_a)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns the pair of images and corresponding cutoff standard deviation
        value at index `idx`.

        Since self.images_a and self.images_b contain paths to the images, you
        should read the images here and normalize the pixels to be between 0 and 1.
        Make sure you transpose the dimensions so that image_a and image_b are of
        shape (c, m, n) instead of the typical (m, n, c), and convert them to
        torch Tensors.

        If you want to use a pair of images that have different dimensions from
        one another, you should resize them to match in this function using
        torchvision.transforms.

        Args
        - idx: int specifying the index at which data should be retrieved
        Returns
        - image_a: Tensor of shape (c, m, n)
        - image_b: Tensor of shape (c, m, n)
        - cutoff: int specifying the cutoff standard deviation corresponding to
          (image_a, image_b) pair

        HINTS:
        - You should use the PIL library to read images
        - You will use self.transform to convert the PIL image to a torch Tensor
        """


        ########################################################################
        # TODO: YOUR CODE HERE
        ########################################################################
        cutoff = 0
        image_a = load_image(self.images_a[idx])
        image_a = image_a.permute(2,0,1)
        image_b = load_image(self.images_b[idx])
        image_b = image_b.permute(2,0,1) 
#         image_a = self.transform(image_a)
#         image_b = self.transform(image_b)
        cutoff = self.cutoffs[idx]
        
        

        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        return image_a, image_b, cutoff
