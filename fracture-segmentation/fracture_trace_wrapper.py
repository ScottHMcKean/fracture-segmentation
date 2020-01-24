# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:08:26 2020

@author: scott.mckean
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import descartes
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiLineString
exec(open('helpers.py').read())
exec(open('FractureTrace.py').read())

# Initialize and load
trace = FractureTrace()
trace.show_figures = True
trace.load_traces('./input/ET_TestWindow_AutoFractures.dxf')
trace.load_masks('./input/ET_TestWindow_Mask.dxf')
trace.scale(scale_m_px = 0.020699)
trace.mask_traces()

# Generate scanlines
trace.scanline_distance_m = 0.5
trace.make_scanlines()
trace.mask_scanlines()
trace.hull_scanlines()
trace.intersect_scanlines()
trace.calc_scanline_stats()

# make rolling segments along scanlines
trace.make_segments()
trace.mask_segments()
trace.hull_segments()
trace.intersect_segments()
trace.calc_segment_stats()

# make rolling windows
trace.make_windows()
trace.mask_windows()
trace.intersect
