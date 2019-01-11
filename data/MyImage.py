import os
import os.path
import random
import math
import errno

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0
        apath ='/data01/kky/KKY/DIV2K/DIV2K_train_LR_bicubicX8_O/X1'
        self.filelist = []
        if not train:
            re = sorted(os.listdir(apath))
            for f in re[800:802]: # in sorted(os.listdir(apath)):
                try:
                    filename = os.path.join(apath, f)
                    misc.imread(filename)
                    self.filelist.append(filename)
                    print(filename)
                   
                except:
                    pass


    def __getitem__(self, idx):
        img_in = misc.imread(self.filelist[idx])
        if len(img_in.shape) == 2:
            img_in = np.expand_dims(img_in, 2)
        h,b,c = img_in.shape
        img_in, img_tar = common.set_channel(img_in, img_in, self.args.n_colors)
        img_tar = misc.imresize(
            img_tar, self.scale[self.idx_scale] * 100, interp='bicubic')

        return common.np2Tensor(img_in, img_tar, self.args.rgb_range)

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

