# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:08:26 2020

@author: scott.mckean
"""
import pandas as pd
import numpy as np  
import math
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm
from scipy import spatial
import geostatspy.GSLIB as GSLIB
import geostatspy.geostats as geostats 
exec(open('GeostatsDataFrame.py').read())

# Initialize Geostats DataFrame
gs_df = GeostatsDataFrame(filepath = './data/sample_geostats_data.csv')
gs_df.set_coord_cols(('x','y'))
gs_df.set_feature_cols(['porosity', 'perm'])
gs_df.z_scale_feats()
gs_df.n_transform_feats()

# Initialize a Variogram Object and perform EDA
vgm = Variogram(gs_df.output)
vgm.convert_azimuth()
vgm.calculate_lags()


vgm.variogram_map('n_porosity', lags = 11, dlag = 50)
