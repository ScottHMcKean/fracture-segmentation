class FractureTrace(object):
    """A class to contain the results of fracture trace analysis from 
    a vector file (shp,dxf,etc., or FractureImage object results"""
    
    show_figures = False
    save_figures = False
    window_width_m = 1
    window_step_increment_m = 0.1
    scanline_distance_m = 0.1
    scale_m_px = 1
    
    def __init__(self):
        []
        
    def list_params(self):
        """ Print a list of object parameters """
        print('show_figures: ' + str(self.show_figures))
        print('save_figures: ' + str(self.save_figures))
        print('window_width_m: ' + str(self.window_width_m))
        print('window_step_increment_m: ' + str(self.window_step_increment_m))
        print('scanline_distance_m: ' + str(self.scanline_distance_m))
        
    def load_traces(self, file_path):
        """ Show image using io.imshow and matplotlib """
        self.traces = gpd.GeoDataFrame(gpd.read_file(file_path))
        
        print('Traces loaded')
        
        if self.show_figures:
            self.traces.plot()
            plt.show()
            
    def scale_traces(self, scale_m_px):
        """ Scale traces """
        self.scale_m_px = scale_m_px
        matrix = [self.scale_m_px, 0, 0, self.scale_m_px, 0, 0]
        self.traces = self.traces.affine_transform(matrix)
        
        print('Overwritting original traces with scaled traces')
        
        if self.show_figures:
            self.traces.plot()
            plt.show()
            
    def make_horizontal_scanlines(self):
        """ Generate horizontal scanlines """
        vert_limits = list(self.traces.total_bounds[i] for i in [1,3])
        horiz_limits = list(self.traces.total_bounds[i] for i in [0,2])
        
        vert_splits = np.arange(
                min(vert_limits) + self.scanline_distance_m/2, 
                max(vert_limits), self.scanline_distance_m
                )
        
        start = list(zip(np.repeat(min(horiz_limits),len(vert_splits)), vert_splits))
        end = list(zip(np.repeat(max(horiz_limits),len(vert_splits)), vert_splits))
        lines = list(zip(start,end))
        names = ['scan_h__' + str(i) for i in np.arange(0,len(lines))+1]
            
        self.scanlines_horizontal = gpd.GeoDataFrame({
                'name': names},
                geometry = gpd.GeoSeries(map(LineString, lines))
                )
        
        print('Horizontal scanlines generated')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            self.scanlines_horizontal.plot(color = 'r', ax=ax)
            plt.show()

    def make_vertical_scanlines(self):
        """ Generate vertical scanlines """
        vert_limits = list(self.traces.total_bounds[i] for i in [1,3])
        horiz_limits = list(self.traces.total_bounds[i] for i in [0,2])
        
        horiz_splits = np.arange(
                min(horiz_limits) + self.scanline_distance_m/2, 
                max(horiz_limits), self.scanline_distance_m
                )
        
        start = list(zip(horiz_splits, np.repeat(min(vert_limits),len(horiz_splits))))
        end = list(zip(horiz_splits, np.repeat(max(vert_limits),len(horiz_splits))))
        lines = list(zip(start,end))
        names = ['scan_v_' + str(i) for i in np.arange(0,len(lines))+1]
            
        self.scanlines_vertical = gpd.GeoDataFrame({
                'name': names},
                geometry =  gpd.GeoSeries(map(LineString, lines))
                )
        
        print('Vertical scanlines generated')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            self.scanlines_vertical.plot(color = 'r', ax=ax)
            plt.show()
            
    def make_scanlines(self):
        self.make_vertical_scanlines()
        self.make_horizontal_scanlines()

    def make_convex_hull(self):
        """ Get convex hull around all traces """
        self.total_convex_hull = self.traces.unary_union.convex_hull
        
        print('Convex hull around traces generated')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            ax.plot(*self.total_convex_hull.exterior.xy, color = 'r')
            plt.show()
            
    def intersect_scanlines_hull(self):
        self.scanlines_horizontal_hulled = gpd.GeoDataFrame(
                {'name' : self.scanlines_horizontal['name']},
                geometry = [self.total_convex_hull.intersection(x) 
                    for x 
                    in self.scanlines_horizontal.geometry]
                )
    
        self.scanlines_vertical_hulled = gpd.GeoDataFrame(
                {'name' : self.scanlines_vertical['name']},
                geometry = [self.total_convex_hull.intersection(x) 
                    for x 
                    in self.scanlines_vertical.geometry]
                )
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            ax.plot(*self.total_convex_hull.exterior.xy, color = 'b')
            self.scanlines_vertical_hulled.plot(color = 'r', ax=ax)
            self.scanlines_horizontal_hulled.plot(color = 'r', ax=ax)
            plt.show()
            
    def intersect_scanlines_traces(self):
        self.intersect_horizontal_scanlines_traces()
        self.intersect_vertical_scanlines_traces()
        
    def intersect_horizontal_scanlines_traces(self):
        self.horiz_scanline_intersections = [
                self.traces.intersection(other = scanline) 
                for scanline
                in self.scanlines_horizontal_hulled.geometry
                ]
        
        self.horiz_scanline_intersected_traces = [
                self.traces[np.invert(intersection.geometry.is_empty)] 
                for intersection
                in self.horiz_scanline_intersections
                ]
        
        self.horiz_scanline_intersected_points = [
                intersection[np.invert(intersection.is_empty)] 
                for intersection
                in self.horiz_scanline_intersections
                ]
        
    def calc_horizontal_scanline_p10(self):
        self.frac_to_frac_lengths = [
                max(points.x) - min(points.x) 
                if len(points) > 0
                else np.nan
                for points
                in self.horiz_scanline_intersected_points
                ]
        
        point_frac_length_tuple = list(
                zip([len(x) for x in self.horiz_scanline_intersected_points], 
                                 self.frac_to_frac_lengths)
                )
        
        self.horiz_scanline_p10 = [x[0]/x[1] if x[1] > 0 else np.nan for x in int_list]