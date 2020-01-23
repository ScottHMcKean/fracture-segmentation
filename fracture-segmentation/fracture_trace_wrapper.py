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

# Generate scanlines
trace.scanline_distance_m = 0.5
trace.make_scanlines()

# Mask traces and scanlines
trace.mask_traces()
trace.mask_scanlines()

# Make convex hull and trim scanline outside of it
trace.make_convex_hull()
trace.intersect_scanlines_hull()

# intersect the scanlines with traces and calculated statistics
trace.intersect_scanlines_traces()
trace.calc_scanline_stats()

# make rolling segments along scanlines
# having issues with the multi line string object...?
trace.make_scanline_segments()
trace.mask_segments()
trace.intersect_segments_hull()
trace.intersect_segments_traces()
trace.calc_segment_stats()
