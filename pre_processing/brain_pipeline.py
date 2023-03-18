from __future__ import print_function
from glob import glob
from skimage import io
from errno import EEXIST
from os.path import isdir
from os import makedirs
import numpy as np
import subprocess
import progressbar

__author__ = "Cesare Catavitello"
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Cesare Catavitello"
__email__ = "cesarec88@gmail.com"
__status__ = "Production"

# np.random.seed(5)  # for reproducibility
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])


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


def normalize(slice_el):
    """

    :param slice_el: image to normalize removing 1% from top and bottom
     of histogram (intensity removal)
    :return: normalized slice
    """

    b = np.percentile(slice_el, 1)
    t = np.percentile(slice_el, 99)
    slice_el = np.clip(slice_el, b, t)
    if np.std(slice_el) == 0:
        return slice_el
    else:
        return (slice_el - np.mean(slice_el)) / np.std(slice_el)


class BrainPipeline(object):
    """
    A class for processing brain scans for one patient
    """

    def __init__(self, path, n4itk=False, n4itk_a