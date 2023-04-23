"""


This code pick randomly an image between all .mha images (picked up randomly) in the specified folder and
convert it into .png image  in accordance to the number of images required.


"""

from __future__ import print_function
import SimpleITK.SimpleITK as sitk
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from glob import glob
from os import makedirs
from os.path import isdir
from errno import EEXIST

__author__ = "Cesare Catavitello"
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Cesare Catavitello"
__email__ = "cesarec88@gmail.com"
__status__ = "Producti