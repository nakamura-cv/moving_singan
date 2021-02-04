import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--image_dir',default='OpticalFlow/movie/moving-gif-processed/moving-gif_longimg/test')
    parser.add_argument('-i', '--image_name', required=True)
    args = parser.parse_args()


    img = cv2.imread('%s/%s' % (args.image_dir, args.image_name))
    for i in range(10):
        cut_image = img[:,i*128:i*128+128,:]
        cut_image = cv2.resize(cut_image, (141,250))
        cv2.imwrite('OpticalFlow/movie/moving-gif-processed/moving-gif_img/00058/%d.png' %i ,cut_image)
        print(i)

