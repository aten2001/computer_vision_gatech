import torch

import numpy as np


def my_1dfilter(signal: torch.FloatTensor,
                kernel: torch.FloatTensor) -> torch.FloatTensor:
    """Filters the signal by the kernel.

    output = signal * kernel where * denotes the cross-correlation function.
    Cross correlation is similar to the convolution operation with difference
    being that in cross-correlation we do not flip the sign of the kernel.

    Reference: 
    - https://mathworld.wolfram.com/Cross-Correlation.html
    - https://mathworld.wolfram.com/Convolution.html

    Note:
    1. The shape of the output should be the same as signal.
    2. You may use zero padding as required. Please do not use any other 
       padding scheme for this function.
    3. Take special care that your function performs the cross-correlation 
       operation as defined even on inputs which are asymmetric.

    Args:
        signal (torch.FloatTensor): input signal. Shape=(N,)
        kernel (torch.FloatTensor): kernel to filter with. Shape=(K,)

    Returns:
        torch.FloatTensor: filtered signal. Shape=(N,)
    """
    filtered_signal = torch.FloatTensor()
    stride=1
    
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    out_shape = signal.shape[0]
    kernel_shape = kernel.shape[0]
    padding = int(( (kernel_shape-1)/2) +0.5) 
    
    signal = torch.nn.functional.pad(signal, (padding,padding), 'constant', 0)
    filtered_signal = torch.zeros(out_shape)
    
    for val in range(out_shape):
        signal_cut = signal[val*stride:val*stride+kernel_shape]
        filtered_signal[val] = torch.sum(torch.mul(signal_cut, kernel))
    filtered_signal= filtered_signal.squeeze(0).squeeze(0)
#     print(filtered_signal.shape)
#     print(out_shape)
#     print("-----------")

    #Only do left padding when padding is an odd number
    if filtered_signal.shape[0]>out_shape:
        filtered_signal = filtered_signal[:out_shape]
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return filtered_signal
