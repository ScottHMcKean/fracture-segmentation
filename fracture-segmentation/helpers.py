def get_size(obj, seen=None):
    """Recursively finds size of objects https://goshippo.com/blog/measure-real-size-any-python-object/"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def convert_geo_list_to_geoseries(geo_list):
    for i in range(0, len(geo_list)):
        if i == 0:
            out = gpd.GeoSeries(geo_list[i])
        else:
            out = out.append(geo_list[i]) 

    return out

def make_vertical_segments(scanline_row, step_increment = 0.1, segment_width = 1):
    # need to adjust the np.arange to step and not divide....
    # need to draw this out. We have x boxes (given by np_arrang and the step increment) and we 
    # want to offset them by a distance (say 1 m - the window width)
    # so we get an index of y_coords and take the first one to the -n one 
    # where n is floor(window_width/step_increment)

    n = int(segment_width/step_increment)
    scanline = scanline_row[1].loc['orig_geom']
    x_coord = np.unique(scanline.xy[0])
    y_coords = np.arange(np.min(scanline.xy[1]), np.max(scanline.xy[1]), step_increment)
    seg_start_point = list(zip(np.repeat(x_coord,len(y_coords[0:-n])), y_coords[0:-n]))
    seg_end_point = list(zip(np.repeat(x_coord,len(y_coords[n:])), y_coords[n:]))
    seg_points = list(zip(seg_start_point,seg_end_point))
    scanline_segments = gpd.GeoSeries(map(LineString, seg_points))
    name = scanline_row[1]['name']
    
    names = [name + '_seg_' + str(i) for i in np.arange(0,len(scanline_segments))+1]
            
    segment_df = gpd.GeoDataFrame({
                'name': names,
                'x_coord': np.repeat(x_coord, len(names)),
                'y_midpoint': (y_coords[0:-n] + y_coords[n:])/2},
                geometry =  scanline_segments
                )
        
    segment_df['orig_length'] = segment_df.length
    segment_df['orig_geom'] = segment_df['geometry']
    
    return segment_df
    
def make_horizontal_segments(scanline_row, step_increment = 0.1, segment_width = 1):
    
    n = int(segment_width/step_increment)
    scanline = scanline_row[1].loc['orig_geom']
    y_coord = np.unique(scanline.xy[1])
    x_coords = np.arange(np.min(scanline.xy[0]), np.max(scanline.xy[0]), step_increment)
    seg_start_point = list(zip(x_coords[0:-n], np.repeat(y_coord,len(x_coords[0:-n]))))
    seg_end_point = list(zip(x_coords[n:],np.repeat(y_coord,len(x_coords[n:]))))
    seg_points = list(zip(seg_start_point,seg_end_point))
    scanline_segments = gpd.GeoSeries(map(LineString, seg_points))
    
    name = scanline_row[1]['name']
    
    names = [name + '_seg_' + str(i) for i in np.arange(0,len(scanline_segments))+1]
            
    segment_df = gpd.GeoDataFrame({
                'name': names,
                'y_coord': np.repeat(y_coord, len(names)),
                'x_midpoint': (x_coords[0:-n] + x_coords[n:])/2},
                geometry =  scanline_segments
                )
        
    segment_df['orig_length'] = segment_df.length
    segment_df['orig_geom'] = segment_df['geometry']
    
    return segment_df