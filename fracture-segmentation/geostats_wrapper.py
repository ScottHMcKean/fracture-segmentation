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
import scipy.spatial.distance as ssd
from scipy import spatial
import geostatspy.GSLIB as GSLIB
import geostatspy.geostats as geostats 
exec(open('GeostatsDataFrame.py').read())
exec(open('Variogram.py').read())

# Initialize Geostats DataFrame
gs_df = GeostatsDataFrame(filepath = './data/sample_geostats_data.csv')
gs_df.set_coord_cols(('x','y'))
gs_df.set_feature_cols(['porosity', 'perm'])
gs_df.z_scale_feats()
gs_df.n_transform_feats()

# Initialize a Variogram Object and calculate lags for everything
vgm = Variogram(gs_df.output, 'n_porosity')
vgm.get_lags_wrapper()

# get omni semivariogram
vgm.calc_omni_variogram()
vgm.omni_variogram.plot('lag_bin','semivariance','scatter')
vgm.write_omni_variogram()

# get azimuth semivariogram for multiple azimuths
for azimuth in [0, 45, 90, 135, 180]:
    vgm.azimuth_cw_from_ns_deg = azimuth
    vgm.calc_azi_variogram()
    vgm.azi_variogram[
            vgm.azi_variogram.azimuth == azimuth
            ].plot('lag_bin','semivariance','scatter')
    plt.show()
    
vgm.write_azi_variogram()

# get variogram map
vgm.make_variogram_map()
vgm.filt_variogram_map(min_points = 3)
vgm.plot_variogram_map()
vgm.plot_npairs_map()
vgm.write_variogram_map()
