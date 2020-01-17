# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:08:26 2020

@author: scott.mckean
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import skimage
import sys
from skimage import io, measure, util
from skimage.restoration import denoise_bilateral
from skimage.feature import canny
from skimage.morphology import binary_closing, square
from skimage.transform import probabilistic_hough_line

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

# edge detection w/ parameters
sample.sig_threshold = 0.6
sample.canny_method = 'standard'
sample.detect_edges()
sample.close_gaps()

# label connected components
sample.label_edges()
sample.count_edges()

# find large edges above a minimum threshold
sample.min_large_edge_px = 50
sample.find_large_edges()

# run the probabilistic hough transform
sample.min_line_length_px = 50
sample.phough_line_gap_px = 10
sample.run_phough_transform()