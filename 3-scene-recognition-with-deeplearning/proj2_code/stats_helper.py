import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then scale to [0,1] before computing
  mean and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  ############################################################################
  # Student code begin
  ############################################################################
  mean =0
  std=0
  directory = os.listdir(dir_name)
  total_data_num =0
  scaler = StandardScaler()
  for i in range(len(directory)):
    dir_names = os.path.join(dir_name, directory[i])
    dir_names = os.listdir(dir_names)
    for j in range(len(dir_names)):
      file_names = os.path.join(dir_name, directory[i], dir_names[j])
      file_names = os.listdir(file_names)
      for k in range(len(file_names)):
        image_file_name = os.path.join(dir_name,directory[i], dir_names[j], file_names[k])
        image = Image.open(image_file_name)
        # convert image to numpy array
        image = ImageOps.grayscale(image)
        pixels = np.asarray(image)
        pixels = pixels.astype('float32')
        # normalize to the range 0-1
        pixels /= 255.0
        pixels = pixels.reshape(-1,1)
        scaler.partial_fit(pixels)

  mean = scaler.mean_
  std = np.sqrt(scaler.var_)
#   mean /=total_data_num
#   std  /=total_data_num
    
  mean = np.array(mean)
  std = np.array(std)
  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
