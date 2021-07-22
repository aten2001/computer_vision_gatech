import copy

import torch

from proj2_code.image_loader import ImageLoader
from proj2_code.my_alexnet import MyAlexNet
# from proj2_code.my_alexnet_quantized import MyAlexNetQuantized
import torch.nn as nn


def quantize_model(float_model: MyAlexNet,
                   train_loader: ImageLoader):
  '''
  Quantize the input model to int8 weights.

  Args:
  -   float_model: model with fp32 weights.
  -   train_loader: training dataset.
  Returns:
  -   quantized_model: equivalent model with int8 weights.
  '''
  quantized_model = MyAlexNet()
  quantized_model = torch.quantization.quantize_dynamic(quantized_model, {torch.nn.Conv2d,nn.MaxPool2d,nn.Linear, nn.ReLU, nn.BatchNorm2d,nn.Flatten}, dtype=torch.qint8)

#   # copy the weights from original model (still floats)
#   quantized_model = MyAlexNetQuantized()
#   quantized_model.cnn_layers = copy.deepcopy(float_model.cnn_layers)
#   quantized_model.fc_layers = copy.deepcopy(float_model.fc_layers)

#   quantized_model = quantized_model.to('cpu')

#   quantized_model.eval()

#   ##############################################################################
#   # Student code begin
#   ##############################################################################
    
# #   quantized_model.fuse_model()
#   quantized_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
#   torch.quantization.prepare(quantized_model, inplace=True)
    
#   for image, label in train_loader:
#     output = quantized_model(image)
    
#   torch.quantization.convert(quantized_model, inplace=True)

#   ##############################################################################
#   # Student code end
#   ##############################################################################

#   quantized_model.eval()

  return quantized_model
