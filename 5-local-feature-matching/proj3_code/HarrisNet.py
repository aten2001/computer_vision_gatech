#!/usr/bin/python3

from torch import nn
import torch
from typing import Tuple


from proj3_code.torch_layer_utils import (
    get_sobel_xy_parameters,
    get_gaussian_kernel,
    ImageGradientsLayer
)



class HarrisNet(nn.Module):
    """
    Implement Harris corner detector (See Szeliski 4.1.1) in pytorch by
    sequentially stacking several layers together.

    Your task is to implement the combination of pytorch module custom layers
    to perform Harris Corner detector.

    Recall that R = det(M) - alpha(trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
    http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.

    You may find the Pytorch function nn.Conv2d() helpful here.
    """

    def __init__(self):
        """
        We Create a nn.Sequential() network, using 5 specific layers (not in this
        order):
          - SecondMomentMatrixLayer: Compute S_xx, S_yy and S_xy, the output is
            a tensor of size (num_image, 3, width, height)
          - ImageGradientsLayer: Compute image gradients Ix Iy. Can be
            approximated by convolving with Sobel filter.
          - NMSLayer: Perform nonmaximum suppression, the output is a tensor of
            size (num_image, 1, width, height)
          - ChannelProductLayer: Compute I_xx, I_yy and I_xy, the output is a
            tensor of size (num_image, 3, width, height)
          - CornerResponseLayer: Compute R matrix, the output is a tensor of
            size (num_image, 1, width, height)

        To help get you started, we give you the ImageGradientsLayer layer to
        compute Ix and Iy. You will need to implement all the other layers.

        Args:
        -   None

        Returns:
        -   None
        """
        super(HarrisNet, self).__init__()

        image_gradients_layer = ImageGradientsLayer()


        # (1) ImageGradientsLayer: Compute image gradients Ix Iy. Can be
        #     approximated by convolving with sobel filter.
        # (2) EigenvalueApproxLayer: Compute S_xx, S_yy and S_xy, the output is
        #     a tensor of size num_image x 3 x width x height
        # (3) CornerResponseLayer: Compute R matrix, the output is a tensor of
        #     size num_image x 1 x width x height
        # (4) NMSLayer: Perform non-maximum suppression, the output is a tensor
        #     of size num_image x 1 x width x height

        layer_1 = ChannelProductLayer()
        layer_2 = SecondMomentMatrixLayer()
        layer_3 = CornerResponseLayer()
        layer_4 = NMSLayer()

        self.net = nn.Sequential(
            image_gradients_layer,
            layer_1,
            layer_2,
            layer_3,
            layer_4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of HarrisNet network. We will only test with 1
        image at a time, and the input image will have a single channel.

        Args:
        -   x: input Tensor of shape (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network,
            (num_image, 1, height, width) tensor

        """
        assert x.dim() == 4, \
            "Input should have 4 dimensions. Was {}".format(x.dim())

        return self.net(x)


class ChannelProductLayer(torch.nn.Module):
    """
    ChannelProductLayer: Compute I_xx, I_yy and I_xy,

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing I_xx, I_yy and I_xy respectively.
    """
    def __init__(self):
        super(ChannelProductLayer, self).__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of the previous layer, which is of size
        (num_image x 2 x width x height) for Ix and Iy.

        Args:
        -   x: input tensor of size (num_image, 2, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for I_xx, I_yy and I_xy.

        HINT: you may find torch.cat(), torch.mul() useful here
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        Ixx = torch.mul(x[:,0,:,:], x[:,0,:,:]).unsqueeze(1)
        Iyy = torch.mul(x[:,1,:,:], x[:,1,:,:]).unsqueeze(1)
        Ixy = torch.mul(x[:,0,:,:], x[:,1,:,:]).unsqueeze(1)
        output = torch.cat((Ixx,Iyy,Ixy), dim=1)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output

class SecondMomentMatrixLayer(torch.nn.Module):
    """
    SecondMomentMatrixLayer: Given a 3-channel image I_xx, I_xy, I_yy, then
    compute S_xx, S_yy and S_xy.

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing S_xx, S_yy and S_xy, respectively

    """
    def __init__(self, ksize: torch.Tensor = 9, sigma: torch.Tensor = 7):
        """
        You may find get_gaussian_kernel() useful. You must use a Gaussian
        kernel with filter size `ksize` and standard deviation `sigma`. After
        you pass the unit tests, feel free to experiment with other values.

        Args:
        -   ksize: single element tensor containing the filter size
        -   sigma: single element tensor containing the standard deviation

        Returns:
        -   None
        """
        super(SecondMomentMatrixLayer, self).__init__()
        self.ksize = ksize
        self.sigma = sigma

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        pad = int((self.ksize-1)/2)
        self.conv2d_gauss = nn.Conv2d(in_channels=1, out_channels =1, kernel_size = self.ksize, bias=False, padding=(pad,pad), padding_mode='zeros')
        self.conv2d_gauss.weight = get_gaussian_kernel(self.ksize, self.sigma)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of previous layer, which is of size
        (num_image, 3, width, height) for I_xx and I_yy and I_xy.

        Args:
        -   x: input tensor of size (num_image, 3, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for S_xx, S_yy and S_xy

        HINT:
        - You can either use your own implementation from project 1 to get the
        Gaussian kernel, OR reimplement it in get_gaussian_kernel().
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        Sxx = self.conv2d_gauss(x[:,0,:,:].unsqueeze(1))
        Syy = self.conv2d_gauss(x[:,1,:,:].unsqueeze(1))
        Sxy = self.conv2d_gauss(x[:,2,:,:].unsqueeze(1))
        output = torch.cat((Sxx,Syy,Sxy), dim=1)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


class CornerResponseLayer(torch.nn.Module):
    """
    Compute R matrix.

    The output is a tensor of size (num_image, channel, height, width),
    represent corner score R

    HINT:
    - For matrix A = [a b;
                      c d],
      det(A) = ad-bc, trace(A) = a+d
    """
    def __init__(self, alpha: int=0.05):
        """
        Don't modify this __init__ function!
        """
        super(CornerResponseLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass to compute corner score R

        Args:
        -   x: input tensor of size (num_image, 3, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        You may find torch.mul() useful here.
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        det = torch.mul(x[:,0,:,:].unsqueeze(1),x[:,1,:,:].unsqueeze(1)) - torch.mul(x[:,2,:,:].unsqueeze(1),x[:,2,:,:].unsqueeze(1))
        trace = x[:,0,:,:].unsqueeze(1) + x[:,1,:,:].unsqueeze(1)
        second_term = self.alpha * (torch.mul(trace, trace))
        output = det - second_term
        
        print("output", output.shape)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


class NMSLayer(torch.nn.Module):
    """
    NMSLayer: Perform non-maximum suppression,

    the output is a tensor of size (num_image, 1, height, width),

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d
    """
    def __init__(self):
        super(NMSLayer, self).__init__()
        pad = int((7-1)/2)
        self.max_pool = nn.MaxPool2d(7, padding=(pad,pad), stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum value in the neighborhood. Binarize the image
        according to locations that are equal to their maximum (i.e. 1 when it
        is equal to the maximum value and 0 otherwise), and return this binary
        image, multiplied with the cornerness response values. We'll be testing
        only 1 image at a time. Input and output will be single channel images.

        Args:
        -   x: input tensor of size (num_image, 1, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        (Potentially) useful functions: nn.MaxPool2d, torch.where(),
        torch.median()
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        x_zeros = torch.zeros(*x.size())
        x = torch.where(x>torch.median(x), x,x_zeros)
        
        b = self.max_pool(x)
        b_ones = torch.ones(*b.size())
        b = torch.where(x==b, b_ones, x_zeros)        
        
        output = torch.mul(b,x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


def get_interest_points(image: torch.Tensor, num_points: int = 4500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to return top most N x,y points with the highest confident corner
    score. Note that the return type should be tensors. Also make sure to
    sort them in descending order of confidence!

    (Potentially) useful functions: torch.nonzero, torch.masked_select,
    torch.argsort

    Args:
    -   image: A tensor of shape (b,c,m,n). We will provide an image of
        (c = 1) for grayscale image.

    Returns:
    -   x: A tensor array of shape (N,) containing x-coordinates of
        interest points
    -   y: A tensor array of shape (N,) containing y-coordinates of
        interest points
    -   confidences: tensor array of dim (N,) containing the
        strength of each interest point
    """

    # We initialize the Harris detector here, you'll need to implement the
    # HarrisNet() class
    harris_detector = HarrisNet()

    # The output of the detector is an R matrix of the same size as image,
    # indicating the corner score of each pixel. After non-maximum suppression,
    # most of R will be 0.
    R = harris_detector(image)
    _,_,_,d = R.shape

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    a = torch.nonzero(R.squeeze(0).squeeze(0), as_tuple=True)
    b = R[torch.nonzero(R, as_tuple=True)]
    
    confidences, ind = torch.sort(b, descending=True)
    y = torch.index_select(a[0],0, ind)
    x = torch.index_select(a[1],0, ind)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    y = y[:num_points]
    x = x[:num_points]
    confidences = confidences[:num_points]
    
    print("x",x.shape)
    if d>50:
        x,y,confidences = remove_border_vals(image, x, y, confidences)

    return x, y, confidences



def remove_border_vals(img, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a 16x16 window around
    that point does not lie completely within the input image.

    Note: maintain the ordering which is input to this function.

    Args:
    -   x: Torch tensor of shape (M,)
    -   y: Torch tensor of shape (M,)
    -   c: Torch tensor of shape (M,)

    Returns:
    -   x: Torch tensor of shape (N,), where N <= M (less than or equal after
        pruning)
    -   y: Torch tensor of shape (N,)
    -   c: Torch tensor of shape (N,)
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    n,m = img.size()[2:]
    border_length = 16
    min_w = border_length
    max_w = m-border_length
    min_h = border_length
    max_h = n-border_length
    
    pruned_ind = []
    for i in range(len(c)):
        if x[i]<min_w or x[i]>max_w or y[i]<min_h or y[i] > max_h:
            continue
        else:
            pruned_ind.append(i)
            
    pruned_ind = torch.LongTensor(pruned_ind)
    x = torch.index_select(x,0, pruned_ind)
    y = torch.index_select(y,0, pruned_ind)
    c = torch.index_select(c,0, pruned_ind)
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return x, y, c


