# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:08:26 2020

@author: scott.mckean
"""
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString

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

# intersect the scanlines with traces
trace.intersect_horizontal_scanlines_traces()

# get scanline stats (p10, )
trace.calc_horizontal_scanline_p10()

combine_geo_list()

def combine_geo_list(geo_list):
   for i in range(0, len(geo_list)):
      if i == 0:
          out = gpd.GeoSeries(geo_list[i])
      else:
          out = out.append(geo_list[i]) 


test = [test.append(points) for points in ]
test = gpd.GeoSeries(trace.horiz_scanline_intersected_points[1])
test = test.append(trace.horiz_scanline_intersected_points[2])

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
