
import numpy as np
import random
import os
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import progressbar
from sklearn.feature_extraction.image import extract_patches_2d

__author__ = "Matteo Causio"

__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Matteo Causio"
__status__ = "Production"

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
np.random.seed(5)

# TODO verify all those function and correct them again
class PatchExtractor(object):
    def __init__(self, patch_size, train_data, num_samples):
        """
        
        :param patch_size: tuple, int, size (in voxels) of patches to extract. Use (33,33) for sequential model
        :param train_data: list, string, list of filepaths to all training data saved as pngs. 
                           images should have shape (5*240,240)
        :param num_samples: int, the number of patches to collect from training data.
        """
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.train_data = train_data
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]

    def find_patches(self, class_num, num_patches):
        """
        
        Helper function for sampling slices with evenly distributed classes
        :param class_num: int, class to sample from choice of {0, 1, 2, 3, 4}.
        :param num_patches: 
        :return: num_samples patches from class 'class_num' randomly selected. 
        """
        # TODO 1 and 3 refers to?
        # '''
        # Helper function for sampling slices with evenly distributed classes
        # INPUT:  (1) list 'training_images': all training images to select from
        #         (2) int 'class_num': class to sample from choice of {0, 1, 2, 3, 4}.
        #         (3) tuple 'patch_size': dimensions of patches to be generated defaults to 65 x 65
        # OUTPUT: (1) num_samples patches from class 'class_num' randomly selected.
        # '''
        h, w = self.patch_size[0], self.patch_size[1]
        patches, labels = [], np.full(num_patches, class_num, 'float')
        print 'Finding patches of class {}...'.format(class_num)

        ct = 0
        while ct < num_patches:
            im_path = random.choice(self.train_data)
            fn = os.path.basename(im_path)
            label = io.imread('Labels/' + fn[:-4] + 'L.png')

            # resample if class_num not in selected slice
            # while len(np.argwhere(label == class_num)) < 10:
            #     im_path = random.choice(self.train_data)
            #     fn = os.path.basename(im_path)
            #     label = io.imread('Labels/' + fn[:-4] + 'L.png')
            if len(np.argwhere(label == class_num)) < 10:
                continue

            # select centerpix (p) and patch (p_ix)
            img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            p = random.choice(np.argwhere(label == class_num))
            p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])

            # resample it patch is empty or too close to edge
            # while patch.shape != (4, h, w) or len(np.unique(patch)) == 1:
            #     p = random.choice(np.argwhere(label == class_num))
            #     p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            #     patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
            if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (h * w):
                continue
            patches.append(patch)
            ct += 1
        return np.array(patches), labels

    def center_n(self, n, patches):
        """
        
        :param n: int, size of center patch to take (square)
        :param patches: list of patches to take subpatch of
        :return: list of center nxn patches.
        """
        sub_patches = []
        for mode in patches:
            subs = np.array(
                [
                    patch[(self.h/2) - (n/2):(self.h/2) + ((n+1)/2), (self.w/2) - (n/2):(self.w/2) + ((n+1)/2)]
                    for patch in mode
                ]
            )
            sub_patches.append(subs)
        return np.array(sub_patches)

# TODO verify if proper correction
    def slice_to_patches(self, filename):
        """
        
        Converts an image to a list of patches with a stride length of 1. Use as input for image prediction.
        :param filename: string, path to image to be converted to patches
        :return: list of patched version of imput image.
        """

        slices = io.imread(filename).astype('float').reshape(5, 240, 240)[:-1]
        plist=[]
        for slice in slices:
            if np.max(slice) != 0:
                slice /= np.max(slice)
            p = extract_patches_2d(slice, (self.h, self.w))
            plist.append(p)
        return np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))

    def patches_by_entropy(self, num_patches):
        """
        
        Finds high-entropy patches based on label, allows net to learn borders more effectively.
        :param num_patches: int, defaults to num_samples, 
                            enter in quantity it using in conjunction with randomly sampled patches.
        :return: list of patches (num_patches, 4, h, w) selected by highest entropy
        """