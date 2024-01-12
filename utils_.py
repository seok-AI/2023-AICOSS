import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from PIL import ImageDraw
import os

def smoothing_label(df):
    for col in list(df.columns)[2:]:
        df.loc[df[col]==0, col] = 0.1
        df.loc[df[col]==1, col] = 0.9
    return df

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd=0.1, 
                 eigval=IMAGENET_PCA['eigval'], 
                 eigvec=IMAGENET_PCA['eigvec']):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        # Create a random vector of 3x1 in the same type as img
        alpha = img.new_empty(3,1).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .matmul(alpha * self.eigval.view(3, 1))

        return img.add(rgb.view(3, 1, 1).expand_as(img))






