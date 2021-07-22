import torch
from torch.quantization import DeQuantStub, QuantStub

from proj2_code.my_alexnet import MyAlexNet
from torchvision.models import alexnet

import torch.nn as nn
from torch.nn import Flatten


class MyAlexNetQuantized(MyAlexNet):
  def __init__(self):
    '''
    Init function to define the layers and loss function.
    '''
    super().__init__()

    self.quant = QuantStub()
    self.dequant = DeQuantStub()
    
    
#     self.alexnet_raw = alexnet(pretrained=True)
#     for param in self.alexnet_raw.parameters():
#         param.requires_grad=False
#     num_features = self.alexnet_raw.classifier[6].in_features
#     self.alexnet_raw.classifier[6]= nn.Linear(num_features, 15)
    
#     self.cnn_layers = self.alexnet_raw.features
#     self.fc_layers = nn.Sequential(Flatten(), 
#                                    self.alexnet_raw.classifier)
#     self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net.

    Hints:
    1. Use the self.quant() and self.dequant() layer on input/output.

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    super().eval()
    model_output = None
    x=self.quant(x)
    model_output = super().forward(x)
#     x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
#     ############################################################################
#     # Student code begin
#     ############################################################################
#     model_output = self.cnn_layers(x)
#     model_output = self.fc_layers(model_output)
    model_output=self.dequant(model_output)
    ############################################################################
    # Student code end
    ############################################################################

    return model_output
