from __future__ import print_function
from glob import glob
from skimage import io
from errno import EEXIST
from os.path import isdir
from os import makedirs
import numpy as np
import subprocess
import progressbar

__author__ = "Cesare Catavitello