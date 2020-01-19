# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:08:26 2020

@author: scott.mckean
"""
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
exec(open('helpers.py').read())
exec(open('FractureTrace.py').read())

# Initialize and load
trace = FractureTrace()
trace.show_figures = True
trace.load_traces('./data/test_edges.dxf')
trace.scale_traces(scale_m_px = 0.001)

# Generate scanlines
trace.make_scanlines()
trace.make_convex_hull()

# trim the outside scanlines
trace.intersect_scanlines_hull()

# intersect the scanlines with traces and calculated statistics
trace.intersect_scanlines_traces()
trace.calc_scanline_stats()

# make rolling segments along scanlines
trace.make_scanline_segments()


## make scanline segments
x_coord = np.unique(scanline.xy[0])
y_coords = np.arange(np.min(scanline.xy[1]), np.max(scanline.xy[1]), trace.window_step_increment_m)
seg_start_point = list(zip(np.repeat(x_coord,len(y_coords[0:-1])), y_coords[0:-1]))
seg_end_point = list(zip(np.repeat(x_coord,len(y_coords[1:])), y_coords[1:]))
seg_points = list(zip(seg_start_point,seg_end_point))
scanline_segments = gpd.GeoSeries(map(LineString, seg_points))

## get p10 of segments
for segment in scanline_segments:
    seg_intersections = trace.traces.intersection(segment)
    points = seg_intersections[np.invert(seg_intersections.is_empty)]
    seg_p10 = len(points)/segment.length
    print(seg_p10)
