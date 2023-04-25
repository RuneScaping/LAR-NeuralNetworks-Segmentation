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
__status__ = "Production"


def mkdir_p(path):
    """
    mkdir -p function, makes folder recursively if required
    :param path:
    :return:
    """
    try:
        makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and isdir(path):
            pass
        else:
            raise


class ImagePngConverter:
    """
    a class to convert an .mha slice into .png image
    to compute a random test with an input image to search for tumor patterns
    """

    def __init__(self, global_counter, path_to_mha=None, how_many_from_one=1, saving_path='./test_data/'):
        if path_to_mha is None:
            raise NameError(' missing .mha path ')
        self.images = []
        for i in range(0, len(path_to_mha)):
            self.images.append(np.array(sitk.GetArrayFromImage(sitk.ReadImage(path_to_mha[i]))))

        mkdir_p(saving_path)
        plt.