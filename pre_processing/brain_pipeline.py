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

    def __init__(self, path, n4itk=False, n4itk_apply=False):
        """

        :param path: path to directory of one patient. Contains following mha files:
        flair, t1, t1c, t2, ground truth (gt)
        :param n4itk:  True to use n4itk normed t1 scans (defaults to True)
        :param n4itk_apply: True to apply and save n4itk filter to t1 and t1c scans for given patient.
        """
        self.path = path
        self.n4itk = n4itk
        self.n4itk_apply = n4itk_apply
        self.modes = ['flair', 't1', 't1c', 't2', 'gt']
        # slices=[[flair x 155], [t1], [t1c], [t2], [gt]], 155 per modality
        self.slices_by_mode, n = self.read_scans()
        # [ [slice1 x 5], [slice2 x 5], ..., [slice155 x 5]]
        self.slices_by_slice = n
        self.normed_slices = self.norm_slices()

    def read_scans(self):
        """
        goes into each modality in patient directory and loads individual scans.
        transforms scans of same slice into strip of 5 images
        """
        print('Loading scans...')
        slices_by_mode = np.zeros((5, 176, 216, 160))
        slices_by_slice = np.zeros((176, 5, 216, 160))
        flair = glob(self.path + '/*Flair*/*.mha')
        t2 = glob(self.path + '/*_T2*/*.mha')
        gt = glob(self.path + '/*more*/*.mha')
        t1s = glob(self.path + '/**/*T1*.mha')
        t1_n4 = glob(self.path + '/*T1*/*_n.mha')
        t1 = [scan for scan in t1s if scan not in t1_n4]
        scans = [flair[0], t1[0], t1[1], t2[0], gt[0]]  # directories to each image (5 total)
        if self.n4itk_apply:
            print('-> Applyling bias correction...')
            for t1_path in t1:
                self.n4itk_norm(t1_path)  # normalize files
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
        elif self.n4itk:
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
        for scan_idx in xrange(5):
            # read each image directory, save to self.slices
            print(io.imread(scans[scan_idx], plugin='simpleitk').astype(float).shape)
            print(scans[scan_idx])
            print('*' * 100)
            try:
                slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
            except:
                continue
        for mode_ix in xrange(slices_by_mode.shape[0]):  # modes 1 thru 5
            for slice_ix in xrange(slices_by_mode.shape[1]):  # slices 1 thru 155
                slices_by_slice[slice_ix][mode_ix] = slices_by_mode[mode_ix][slice_ix]  # reshape by slice
        return slices_by_mode, slices_by_slice

    def norm_slices(self):
        """
        normalizes each slice in self.slices_by_slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
        if n4itk == True, will apply n4itk bias correction to T1 and T1c images
        """
        print('Normalizing slices...')
        normed_slices = np.zeros((176, 5, 216, 160))
        for slice_ix in xrange(176):
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in xrange(4):
                normed_slices[slice_ix][mode_ix] = normalize(self.slices_by_slice[slice_ix][mode_ix])
        print ('Done.')
        return normed_slices

    def save_patient(self, reg_norm_n4, patient_num):
        """
        saves png in Norm_PNG directory for normed, Training_PNG for reg
        :param reg_norm_n4:  'reg' for original images, 'norm' normalized