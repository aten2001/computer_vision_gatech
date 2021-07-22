"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_1D_Gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    """Creates a 1D Gaussian kernel using the specified standard deviation.

    Note: ensure that the value of the kernel sums to 1.

    Args:
        standard_deviation (float): standard deviation of the gaussian

    Returns:
        torch.FloatTensor: required kernel as a row vector
    """

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    
    k = int(4*standard_deviation +1)
    k = torch.as_tensor(k)
    mu = torch.tensor(k/2)
    mu = mu.type(torch.FloatTensor)
    sigma = torch.as_tensor(standard_deviation)
#     Z = 1.0/((torch.sqrt(torch.tensor(2*math.pi))*sigma))
    kernel = [torch.exp(-(i-mu)**2/(2*sigma**2)) for i in range(k)]
    kernel = torch.FloatTensor(kernel)
    
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return kernel/torch.sum(kernel)


def create_2D_Gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    """Creates a 2D Gaussian kernel using the specified standard deviation in
    each dimension, and no cross-correlation between dimensions,

    i.e. 
    sigma_matrix = [standard_deviation^2    0
                    0                       standard_deviation^2]


    The kernel should have:
    - shape (k, k) where k = standard_deviation * 4 + 1
    - mean = floor(k / 2)
    - values that sum to 1

    Args:
        standard_deviation (float): the standard deviation along a dimension

    Returns:
        torch.FloatTensor: 2D Gaussian kernel

    HINT:
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      vectors drawn from 1D Gaussian distributions.
    """
    
    

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    k = int(4*standard_deviation +1)
    k = torch.as_tensor(k)
    mu = torch.tensor(k/2)
    mu = mu.type(torch.FloatTensor)
    sigma = torch.as_tensor(standard_deviation)
#     Z = 1.0/((torch.sqrt(torch.tensor(2*math.pi))*sigma))
    kernel = [torch.exp(-(i-mu)**2/(2*sigma**2)) for i in range(k)]
    kernel_1d = torch.FloatTensor(kernel)
    kernel_2d = torch.ger(kernel_1d, kernel_1d)
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return kernel_2d/kernel_2d.sum()


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the HybridImageModel class.
        """
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_standarddeviation: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff standard deviation.

        PyTorch requires the kernel to be of a particular shape in order to apply
        it to an image. Specifically, the kernel needs to be of shape (c, 1, k, k)
        where c is the # channels in the image. Start by getting a 2D Gaussian
        kernel using your implementation from Part 1, which will be of shape
        (k, k). Then, let's say you have an RGB image, you will need to turn this
        into a Tensor of shape (3, 1, k, k) by stacking the Gaussian kernel 3
        times.

        Args
        - cutoff_standarddeviation: int specifying the cutoff standard deviation
        Returns
        - kernel: Tensor of shape (c, 1, k, k) where c is # channels

        HINTS:
        - You will use the create_Gaussian_kernel() function from part1.py in this
          function.
        - Since the # channels may differ across each image in the dataset, make
          sure you don't hardcode the dimensions you reshape the kernel to. There
          is a variable defined in this class to give you channel information.
        - You can use torch.reshape() to change the dimensions of the tensor.
        - You can use torch's repeat() to repeat a tensor along specified axes.
        """
        kernel = torch.Tensor()

        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################
        c=3
        k = int(4*cutoff_standarddeviation +1)
        k = torch.as_tensor(k)
        mu = torch.tensor(k/2)
        mu = mu.type(torch.FloatTensor)
        sigma = torch.as_tensor(cutoff_standarddeviation)
    #     Z = 1.0/((torch.sqrt(torch.tensor(2*math.pi))*sigma))
        kernel = [torch.exp(-(i-mu)**2/(2*sigma**2)) for i in range(k)]
        kernel_1d = torch.FloatTensor(kernel)
        kernel_2d = torch.ger(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d/kernel_2d.sum()
        
        
        kernel = kernel_2d.expand(3, *kernel_2d.size()).unsqueeze(1)

        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return kernel

    def low_pass(self, x, kernel):
        """
        Applies low pass filter to the input image.

        Args:
        - x: Tensor of shape (b, c, m, n) where b is batch size
        - kernel: low pass filter to be applied to the image
        Returns:
        - filtered_image: Tensor of shape (b, c, m, n)

        HINT:
        - You should use the 2d convolution operator from torch.nn.functional.
        - Make sure to pad the image appropriately (it's a parameter to the
          convolution function you should use here!).
        - Pass self.n_channels as the value to the "groups" parameter of the
          convolution function. This represents the # of channels that the filter
          will be applied to.
        """



        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################
        
        stride=1
        N,C,H,W = x.shape 
    #     filter = filter.expand(C, *filter.size())
        C,N, HH, WW = kernel.shape
        filtered_image = torch.Tensor()
        
        h_pad = int(((HH-1)/2) +0.5) 
        w_pad = int(((WW-1)/2) +0.5) 
        
        filtered_image = torch.nn.functional.conv2d(x, kernel, padding=(h_pad, w_pad), groups=3)

        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return filtered_image

    def forward(self, image1, image2, cutoff_standarddeviation):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the hybrid
        image.

        Args
        - image1: Tensor of shape (b, m, n, c)
        - image2: Tensor of shape (b, m, n, c)
        - cutoff_standarddeviation: Tensor of shape (b)
        Returns:
        - low_frequencies: Tensor of shape (b, m, n, c)
        - high_frequencies: Tensor of shape (b, m, n, c)
        - hybrid_image: Tensor of shape (b, m, n, c)

        HINTS:
        - You will use the get_kernel() function and your low_pass() function in
          this function.
        - Similar to Part 1, you can get just the high frequency content of an
          image by removing its low frequency content.
        - Don't forget to make sure to clip the pixel values >=0 and <=1. You can
          use torch.clamp().
        - If you want to use images with different dimensions, you should resize
          them in the HybridImageDataset class using torchvision.transforms.
        """
        self.n_channels = image1.shape[1]

        low_frequencies = torch.Tensor()
        high_frequencies = torch.Tensor()
        hybrid_image = torch.Tensor()

        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################
        kernel = self.get_kernel(cutoff_standarddeviation)
        low_frequencies  = self.low_pass(image1, kernel)
        low_pass2 = self.low_pass(image2, kernel)
        high_frequencies = image2-low_pass2
        hybrid_image = low_frequencies+ high_frequencies

        hybrid_image = torch.clamp(hybrid_image, min=0, max=1)

        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return low_frequencies, high_frequencies, hybrid_image
