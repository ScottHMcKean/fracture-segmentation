# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:08:26 2020

@author: scott.mckean
"""
import matplotlib.pyplot as plt
import skimage
from skimage import io, measure, util
from skimage.restoration import denoise_bilateral
from skimage.feature import canny
from skimage.morphology import binary_closing, square

# load and set output flags
sample = FractureSegment('./data/handsample1.jpg')
sample.show_img()
sample.list_params()
sample.show_figures = True
sample.save_figures = True

# Denoising filter w/ parameters
sample.sig_color = 0.1
sample.sig_spatial = 0.1
sample.pixel_window = 3
sample.denoise()

# Edge detection w/ parameters
sample.sig_threshold = 0.6
sample.canny_method = 'standard'
sample.detect_edges()
sample.close_gaps()

# label connected components
sample.label_edges()
sample.count_edges()

# find large edges
sample.min_edge_px = 200
sample.find_large_edges()
