import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d,Flatten, Linear, Dropout

class SimpleNetAblation(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    '''
    super(SimpleNetAblation, self).__init__()

    ############################################################################
    # Student code begin
    ############################################################################

    self.cnn_layers = nn.Sequential(
                                    Conv2d(1,10,5), 
                                    ReLU(),
                                    MaxPool2d(kernel_size=3),
                                    Conv2d(10,20,5),
                                    ReLU(),
                                    MaxPool2d(kernel_size=3)
                               )  # conv2d and supporting layers here
    
    self.fc_layers = nn.Sequential(Flatten(),
                                   Linear(500,200),
                                   ReLU(),
                                   Linear(200,15))  # linear and supporting layers here
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    ############################################################################
    # Student code end
    ############################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ###########################################################################
    # Student code begin
    ############################################################################
    x=self.cnn_layers(x)
    model_output = self.fc_layers(x)

    ############################################################################
    # Student code end
    ############################################################################

    return model_output
