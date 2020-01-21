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

def make_vertical_segments(scanline, step_increment = 0.1):
    x_coord = np.unique(scanline.xy[0])
    y_coords = np.arange(np.min(scanline.xy[1]), np.max(scanline.xy[1]), step_increment)
    seg_start_point = list(zip(np.repeat(x_coord,len(y_coords[0:-1])), y_coords[0:-1]))
    seg_end_point = list(zip(np.repeat(x_coord,len(y_coords[1:])), y_coords[1:]))
    seg_points = list(zip(seg_start_point,seg_end_point))
    scanline_segments = gpd.GeoSeries(map(LineString, seg_points))
    return MultiLineString(list(scanline_segments))
    
def make_horizontal_segments(scanline, step_increment = 0.1):
    y_coord = np.unique(scanline.xy[1])
    x_coords = np.arange(np.min(scanline.xy[0]), np.max(scanline.xy[0]), step_increment)
    seg_start_point = list(zip(x_coords[0:-1], np.repeat(y_coord,len(x_coords[0:-1]))))
    seg_end_point = list(zip(x_coords[1:],np.repeat(x_coords,len(y_coords[1:]))))
    seg_points = list(zip(seg_start_point,seg_end_point))
    scanline_segments = gpd.GeoSeries(map(LineString, seg_points))
    return MultiLineString(list(scanline_segments))
