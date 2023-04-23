"""


This code pick randomly an image between all .mha images (picked up randomly) in the specified folder and
convert it into .png image  in accordance to the number of images required.


"""

from __future__ import print_function
import SimpleITK.SimpleITK as sitk
import matplotlib.pyplot as plt
import rando