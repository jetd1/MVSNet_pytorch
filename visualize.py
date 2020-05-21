#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Depth map visualization.
"""

import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from datasets.data_io import read_pfm as load_pfm


def cvt_depth(depth_path):
    assert depth_path.endswith('npy')
    depth_image = np.load(depth_path)
    depth_image = np.squeeze(depth_image)
    print('value range: ', depth_image.min(), depth_image.max())
    # plt.imshow(depth_image, 'rainbow')
    # plt.show()
    plt.imsave(depth_path[:-3] + 'png', depth_image, cmap='rainbow')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_path')
    args = parser.parse_args()
    depth_path = args.depth_path
    if os.path.isdir(depth_path):
        for filename in os.listdir(depth_path):
            if filename.endswith('pfm'):
                cvt_depth(os.path.join(depth_path, filename))
    else:
        cvt_depth(depth_path)
