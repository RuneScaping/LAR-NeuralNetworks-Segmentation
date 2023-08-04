
import numpy as np
import progressbar
from glob import glob
from skimage import io
from os import makedirs
from os.path import isdir
from errno import EEXIST
import subprocess
import argparse

np.random.seed(5) # for reproducibility
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



class ImagePreProcessing(object):
    """
    A class for pre process brain scans for one patient
    :param: path: string, path to directory of one patient. Contains following mha files:
                  flair, t1, t1c, t2,gound truth (gt)
    :param: n4itk: boolean, to specify to use n4itk normed t1 scans (default to True)
    :param: n4itk_apply: boolean, to apply and save n4itk filter to t1 and t1c scans for given patient.            
    """

    def __init__(self, path, n4itk=False, n4itk_apply=False):
        self.path = path
        self.n4itk = n4itk
        self.n4itk_apply = n4itk_apply
        self.modes = ['flair', 't1', 't1c', 't2', 'gt']
        self.slices_by_mode, self.slices_by_slice, self.normed_slices = None, None, None
        self.read_scans()
        self.norm_slices()

    def read_scans(self):
        """
        goes into each modality in patient directory and loads individuals scans.
        transforms scans of same slice into strip of 5 images
        :return: slice_by_slice: 
                 slice_by_mode: 
        
        """
        print 'Loading scans...'
        slices_by_mode = np.zeros((5, 176, 216, 160))
        slice_by_slice = np.zeros((176, 5, 216, 160))
        flair = glob(self.path+'/*Flair*/*.mha')
        t2 = glob(self.path+'/*_T2*/*.mha')
        gt = glob(self.path+'/*more*/*.mha')
        t1s = glob(self.path+'/**/*T1*.mha')
        t1_n4 = glob(self.path+'/*T1*/*_n.mha')
        t1 = [scan for scan in t1s if scan not in t1_n4]
        scans = [flair[0], t1[0], t1[1], t2[0], gt[0]]  # directories to each image (5 total)
        if self.n4itk_apply:
            print '-> Applyling bias correction...'
            for t1_path in t1:
                self.n4itk_norm(t1_path)  # normalize files
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
        elif self.n4itk:
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
        for scan_idx, scan_el in enumerate(scans):  # read each image directory, save to self.slices
            slices_by_mode[scan_idx] = io.imread(scan_el, plugin='simpleitk').astype(float)
        for mode_ix in xrange(slices_by_mode.shape[0]):  # modes 1 thru 5 or 4
            for slice_ix in xrange(slices_by_mode.shape[1]):  # slices 1 thru 155
                slice_by_slice[slice_ix][mode_ix] = slices_by_mode[mode_ix][slice_ix]  # reshape by slice
        self.slices_by_slice = slice_by_slice
        self.slices_by_mode = slices_by_mode

    def norm_slices(self):
        """
        normalizes each slice in self.slice_by_slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel  intensities
        if n4itk == True, will apply apply bias correction to T1 and T1c images
        :return: normed_slices:
        """
        print 'Normalizing slices...'
        normed_slices = np.zeros((176, 5, 216, 160))
        for slice_ix in xrange(176):
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in xrange(4):
                normed_slices[slice_ix][mode_ix] = self._normalize(self.slices_by_slice[slice_ix][mode_ix])
        print 'Done.'
        self.normed_slices = normed_slices

    def _normalize(self, passed_slice):
        """  
        normalize slices
        :param slice: a single slice of any modality (excluding gt).
                      all index of modality assoc with slice: 
                      (0=flair, 1=t1, 2=t1c, 3=t2).
        :return: normalized slice
        """
        b = np.percentile(passed_slice, 99)