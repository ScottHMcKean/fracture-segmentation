class FractureTrace(object):
    """A class to contain the results of fracture trace analysis from 
    a vector file (shp,dxf,etc., or FractureImage object results"""
    
    show_figures = False
    save_figures = False
    limit_direction_to = None  #'horizontal', 'vertical', or 'None'
    segment_width_m = 1
    segment_step_increment_m = 0.2
    scanline_distance_m = 0.5
    scale_m_px = 1
    window_width_m = 1
    window_step_increment_m = 0.2
    
    def __init__(self):
        []
        
    def list_params(self):
        """ Print a list of object parameters """
        print('show_figures: ' + str(self.show_figures))
        print('save_figures: ' + str(self.save_figures))
        print('segment_width_m: ' + str(self.window_width_m))
        print('segment_step_increment_m: ' + str(self.window_step_increment_m))
        print('scanline_distance_m: ' + str(self.scanline_distance_m))
        
    def load_traces(self, file_path):
        """ Show image using io.imshow and matplotlib """
        self.traces = gpd.GeoDataFrame(gpd.read_file(file_path))
        
        print('Traces loaded')
        
        if self.show_figures:
            self.traces.plot()
            plt.show()
            
    def load_masks(self, file_path):
        """ Loads mask, selects only polygons """
        self.masks = gpd.GeoDataFrame(gpd.read_file(file_path))
        self.masks = self.masks[self.masks.geometry.geom_type == 'Polygon']
        self.masks = self.masks.reset_index()
        
        print('Masks loaded')
        
        if self.show_figures:
            self.masks.plot()
            plt.show()
            
    def scale(self, scale_m_px):
        """ Scale traces """
        self.scale_m_px = scale_m_px
        matrix = [self.scale_m_px, 0, 0, self.scale_m_px, 0, 0]
        self.traces = self.traces.affine_transform(matrix)
        print('Scaling and overwritting traces')
        
        if hasattr(self, 'masks'):
            self.masks = self.masks.affine_transform(matrix)
            print('Scaling and overwritting masks')
        
        if self.show_figures:
            self.traces.plot(color = 'k')
            if hasattr(self, 'masks'): self.masks.plot(color = 'r')
            plt.show()
            
    def mask_traces(self):
        """ Mask traces """
        self.traces_orig = self.traces
        
        for mask in self.masks:
            trace_diff = self.traces.difference(mask)
            self.traces = trace_diff[~trace_diff.is_empty] 
        
        print('Masking traces (saved & overwritten)')
        
        if self.show_figures:
                fig, ax = plt.subplots(1, 1)
                self.traces_orig.plot(color = 'k', ax=ax, alpha = 0.5)
                self.traces.plot(color = 'r', ax=ax)
                for mask in self.masks:
                    ax.plot(*mask.exterior.xy, color = 'b')
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
        names = ['scan_h_' + str(i) for i in np.arange(0,len(lines))+1]
            
        self.horizontal_scanlines = gpd.GeoDataFrame({
                'name': names,
                'y_coord': vert_splits},
                geometry = gpd.GeoSeries(map(LineString, lines))
                )
        
        self.horizontal_scanlines['orig_length'] = self.horizontal_scanlines.length
        self.horizontal_scanlines['orig_geom'] = self.horizontal_scanlines['geometry']
        
        print('Horizontal scanlines generated')

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
        names = ['scan_v' + str(i) for i in np.arange(0,len(lines))+1]
            
        self.vertical_scanlines = gpd.GeoDataFrame({
                'name': names,
                'x_coord': horiz_splits},
                geometry =  gpd.GeoSeries(map(LineString, lines))
                )
        
        self.vertical_scanlines['orig_length'] = self.vertical_scanlines.length
        self.vertical_scanlines['orig_geom'] = self.vertical_scanlines['geometry']
        
        print('Vertical scanlines generated')
            
    def make_scanlines(self):
        if self.limit_direction_to != 'vertical':
            self.make_horizontal_scanlines()
            
        if self.limit_direction_to != 'horizontal':
            self.make_vertical_scanlines()
  
        print('Scanlines intersected with convex hull')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            
            if self.limit_direction_to != 'vertical':
                self.horizontal_scanlines.plot(color = 'b', ax=ax)
                
            if self.limit_direction_to != 'horizontal':
                self.vertical_scanlines.plot(color = 'r', ax=ax)
            
            plt.show()
        
    def mask_horizontal_scanlines(self):
        self.horizontal_scanlines_orig = self.horizontal_scanlines
        
        for mask in self.masks: 
            for (i,line) in enumerate(self.horizontal_scanlines.geometry):
                self.horizontal_scanlines.geometry[i] = self.horizontal_scanlines.geometry[i].difference(mask)
        
        self.horizontal_scanlines['masked_geom'] = self.horizontal_scanlines.geometry
        self.horizontal_scanlines['masked_length'] = self.horizontal_scanlines.length
        
        print('Masking horizontal scanlines (saved & overwritten)')
                
    def mask_vertical_scanlines(self):
        self.vertical_scanlines_orig = self.vertical_scanlines
        
        for mask in self.masks: 
            for (i,line) in enumerate(self.vertical_scanlines.geometry):
                self.vertical_scanlines.geometry[i] = self.vertical_scanlines.geometry[i].difference(mask)
        
        self.vertical_scanlines['masked_geom'] = self.vertical_scanlines.geometry
        self.vertical_scanlines['masked_length'] = self.vertical_scanlines.length
        
        print('Masking vertical scanlines (saved & overwritten)')
        
    def mask_scanlines(self):
        if self.limit_direction_to != 'vertical':
            self.mask_horizontal_scanlines()
            
        if self.limit_direction_to != 'horizontal':
            self.mask_vertical_scanlines()
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            
            if self.limit_direction_to != 'vertical':
                (self
                 .horizontal_scanlines_orig[~self.horizontal_scanlines_orig.is_empty]
                 .plot(color = 'k', ax=ax, alpha = 0.5)
                 )
                (self
                 .horizontal_scanlines[~self.horizontal_scanlines.is_empty]
                 .plot(color = 'r', ax=ax)
                 )
                
            if self.limit_direction_to != 'horizontal':
                (self
                 .vertical_scanlines_orig[~self.vertical_scanlines_orig.is_empty]
                 .plot(color = 'k', ax=ax, alpha = 0.5)
                 )
                (self
                 .vertical_scanlines[~self.vertical_scanlines.is_empty]
                 .plot(color = 'b', ax=ax)
                 )
            
            for mask in self.masks:
                    ax.plot(*mask.exterior.xy, color = 'k')
            
            plt.show()

    def hull_horizontal_scanlines(self):
        self.horizontal_scanlines.geometry = [
                self.traces.unary_union.convex_hull.intersection(x) 
                    for x 
                    in self.horizontal_scanlines.geometry]
    
        self.horizontal_scanlines['hull_trimmed'] = self.horizontal_scanlines.geometry
        self.horizontal_scanlines['trimmed_length'] = self.horizontal_scanlines.length
     
    def hull_vertical_scanlines(self):
        self.vertical_scanlines.geometry = [
                self.traces.unary_union.convex_hull.intersection(x) 
                    for x 
                    in self.vertical_scanlines.geometry]
            
        self.vertical_scanlines['hull_trimmed'] = self.vertical_scanlines.geometry
        self.vertical_scanlines['trimmed_length'] = self.vertical_scanlines.length
        
    def hull_scanlines(self):
        if self.limit_direction_to != 'vertical':
            self.hull_horizontal_scanlines()
            
        if self.limit_direction_to != 'horizontal':
            self.hull_vertical_scanlines()
  
        print('Scanlines intersected with convex hull')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            ax.plot(*self.traces.unary_union.convex_hull.exterior.xy, color = 'k')
            
            if self.limit_direction_to != 'vertical':
                (self
                 .horizontal_scanlines[~self.horizontal_scanlines.is_empty]
                 .plot(color = 'r', ax=ax))
                
            if self.limit_direction_to != 'horizontal':
                (self
                 .vertical_scanlines[~self.vertical_scanlines.is_empty]
                 .plot(color = 'b', ax=ax))
                
            plt.show()
                
    def intersect_horizontal_scanlines(self):
        self.horiz_scanline_intersections = [
                self.traces.intersection(other = scanline) 
                for scanline
                in self.horizontal_scanlines.geometry
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
        
        print('Horizontal scanlines and traces intersected')
    
    def intersect_vertical_scanlines(self):
        self.vert_scanline_intersections = [
                self.traces.intersection(other = scanline) 
                for scanline
                in self.vertical_scanlines.geometry
                ]
        
        self.vert_scanline_intersected_traces = [
                self.traces[np.invert(intersection.geometry.is_empty)] 
                for intersection
                in self.vert_scanline_intersections
                ]
        
        self.vert_scanline_intersected_points = [
                intersection[np.invert(intersection.is_empty)] 
                for intersection
                in self.vert_scanline_intersections
                ]
        
        print('Vertical scanlines and traces intersected')
    
    def intersect_scanlines(self):
        if self.limit_direction_to != 'vertical':
            self.intersect_horizontal_scanlines()
            
        if self.limit_direction_to != 'horizontal':
            self.intersect_vertical_scanlines()
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax, alpha=0.5)
            
            if self.limit_direction_to != 'vertical':
                (self
                 .horizontal_scanlines[~self.horizontal_scanlines.is_empty]
                 .plot(color = 'k', ax=ax, alpha = 0.5)
                 )
                
                points_series = convert_geo_list_to_geoseries(
                        self.horiz_scanline_intersected_points
                        )
                
                points_series.plot(color = 'r', ax=ax, markersize=10)
            
            if self.limit_direction_to != 'horizontal':
                (self
                 .vertical_scanlines[~self.vertical_scanlines.is_empty]
                 .plot(color = 'k', ax=ax, alpha = 0.5)
                 )
                
                points_series = convert_geo_list_to_geoseries(
                        self.vert_scanline_intersected_points
                        )
                
                points_series.plot(color = 'b', ax=ax, markersize=10)
            
            plt.show()
    
    def calc_horizontal_scanline_stats(self):
        
        self.horizontal_scanlines['frac_to_frac_length'] = [
                max(points.x) - min(points.x) 
                if len(points) > 0
                else np.nan
                for points
                in self.horiz_scanline_intersected_points
                ]
        
        point_frac_list = list(
                zip([len(point) 
                for point
                in self.horiz_scanline_intersected_points], 
                self.horizontal_scanlines['frac_to_frac_length'])
                )
        
        self.horizontal_scanlines['p10_frac'] = (
                [x[0]/x[1] if x[1] > 0 else np.nan 
                 for x in point_frac_list]
                )
        
        point_trimmed_list = list(
                zip([len(point) 
                for point
                in self.horiz_scanline_intersected_points], 
                self.horizontal_scanlines['trimmed_length'])
                )
        
        self.horizontal_scanlines['p10_trimmed'] = (
                [x[0]/x[1] if x[1] > 0 else np.nan 
                 for x in point_trimmed_list]
                )
       
        print('Horizontal scanline stats calculated')
        
    def calc_vertical_scanline_stats(self):
        
        self.vertical_scanlines['frac_to_frac_length'] = [
                max(points.y) - min(points.y) 
                if len(points) > 0
                else np.nan
                for points
                in self.vert_scanline_intersected_points
                ]
        
        point_frac_list = list(
                zip([len(point) 
                for point 
                in self.vert_scanline_intersected_points], 
                self.vertical_scanlines['frac_to_frac_length'])
                )
        
        self.vertical_scanlines['p10_frac'] = (
                [x[0]/x[1] if x[1] > 0 else np.nan 
                 for x in point_frac_list]
                )
        
        point_trimmed_list = list(
                zip([len(point) 
                for point 
                in self.vert_scanline_intersected_points], 
                self.vertical_scanlines['trimmed_length'])
                )
        
        self.vertical_scanlines['p10_trimmed'] = (
                [x[0]/x[1] if x[1] > 0 else np.nan 
                 for x in point_trimmed_list]
                )
        
        print('Vertical scanline stats calculated')
        
    def calc_scanline_stats(self):
        if self.limit_direction_to != 'vertical':
            self.calc_horizontal_scanline_stats()
            
        if self.limit_direction_to != 'horizontal':
            self.calc_vertical_scanline_stats()
        
    def write_scanline_tables(self):
        if self.limit_direction_to != 'vertical':
            self.horizontal_scanlines.to_csv('horizontal_scanlines.csv')
            
        if self.limit_direction_to != 'horizontal':
            self.vertical_scanlines.to_csv('vertical_scanlines.csv')
        
    def make_vertical_segments(self):
        self.vertical_segments = pd.concat(
                [make_vertical_segments(
                        x, 
                        step_increment = self.segment_step_increment_m,
                        segment_width = self.segment_width_m
                        )
                for x
                in self.vertical_scanlines.iterrows()]
                ).reset_index()
                
        print('Vertical segments generated')
    
    def make_horizontal_segments(self):
        self.horizontal_segments = pd.concat(
                [make_horizontal_segments(
                        x, 
                        step_increment = self.segment_step_increment_m,
                        segment_width = self.segment_width_m
                        )
                for x
                in self.horizontal_scanlines.iterrows()]
                ).reset_index()
                
        print('Horizontal segments generated')
        
    def make_segments(self):
        if self.limit_direction_to != 'vertical':
            self.make_horizontal_segments()
            
        if self.limit_direction_to != 'horizontal':
            self.make_vertical_segments()
        
    def mask_horizontal_segments(self):
        self.horizontal_segments_orig = self.horizontal_segments
        
        for mask in self.masks: 
            for (i,line) in enumerate(self.horizontal_segments.geometry):
                self.horizontal_segments.geometry[i] = self.horizontal_segments.geometry[i].difference(mask)
        
        self.horizontal_segments['masked_geom'] = self.horizontal_segments.geometry
        self.horizontal_segments['masked_length'] = self.horizontal_segments.length
        
        print('Masking horizontal segments (saved & overwritten)')
                
    def mask_vertical_segments(self):
        self.vertical_segments_orig = self.vertical_segments
        
        for mask in self.masks: 
            for (i,line) in enumerate(self.vertical_segments.geometry):
                self.vertical_segments.geometry[i] = self.vertical_segments.geometry[i].difference(mask)
        
        self.vertical_segments['masked_geom'] = self.vertical_segments.geometry
        self.vertical_segments['masked_length'] = self.vertical_segments.length
        
        print('Masking vertical segments (saved & overwritten)')
  
    def mask_segments(self):
        if self.limit_direction_to != 'vertical':
            self.mask_horizontal_segments()
            
        if self.limit_direction_to != 'horizontal':
            self.mask_vertical_segments()
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            if self.limit_direction_to != 'vertical':
                (self
                 .horizontal_segments_orig[~self.horizontal_segments_orig.is_empty]
                 .plot(color = 'k', ax=ax, alpha = 0.5)
                 )
                (self
                 .horizontal_segments[~self.horizontal_segments.is_empty]
                 .plot(color = 'r', ax=ax)
                 )
            if self.limit_direction_to != 'horizontal':
                (self
                 .vertical_segments_orig[~self.vertical_segments_orig.is_empty]
                 .plot(color = 'k', ax=ax, alpha = 0.5)
                 )
                (self
                 .vertical_segments[~self.vertical_segments.is_empty]
                 .plot(color = 'b', ax=ax)
                 )
            
            for mask in self.masks:
                ax.plot(*mask.exterior.xy, color = 'k')
            
            plt.show()
            
    def intersect_horizontal_segments(self):
        self.horiz_segment_intersections = [
                self.traces.intersection(other = segment) 
                for segment
                in self.horizontal_segments.geometry
                ]
        
        self.horiz_segment_intersected_traces = [
                self.traces[np.invert(intersection.geometry.is_empty)] 
                for intersection
                in self.horiz_segment_intersections
                ]
        
        self.horiz_segment_intersected_points = [
                intersection[np.invert(intersection.is_empty)] 
                for intersection
                in self.horiz_segment_intersections
                ]
        
        print('Horizontal segments and traces intersected')
    
    def intersect_vertical_segments(self):
        self.vert_segment_intersections = [
                self.traces.intersection(other = segment) 
                for segment
                in self.vertical_segments.geometry
                ]
        
        self.vert_segment_intersected_traces = [
                self.traces[np.invert(intersection.geometry.is_empty)] 
                for intersection
                in self.vert_segment_intersections
                ]
        
        self.vert_segment_intersected_points = [
                intersection[np.invert(intersection.is_empty)] 
                for intersection
                in self.vert_segment_intersections
                ]
        
        print('Vertical segments and traces intersected')
    
    def intersect_segments(self):
        if self.limit_direction_to != 'vertical':
            self.intersect_horizontal_segments()
            
        if self.limit_direction_to != 'horizontal':
            self.intersect_vertical_segments()
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax, alpha=0.5)
            
            if self.limit_direction_to != 'vertical':
                (self
                 .horizontal_segments[~self.horizontal_segments.is_empty]
                 .plot(color = 'k', ax=ax, alpha = 0.5)
                 )
                horiz_points_series = convert_geo_list_to_geoseries(
                        self.horiz_segment_intersected_points
                        )
                horiz_points_series.plot(color = 'r', ax=ax, markersize=10)
            
            if self.limit_direction_to != 'horizontal':
                (self
                 .vertical_segments[~self.vertical_segments.is_empty]
                 .plot(color = 'k', ax=ax, alpha = 0.5)
                 )
                vert_points_series = convert_geo_list_to_geoseries(
                        self.vert_segment_intersected_points
                        )
                vert_points_series.plot(color = 'b', ax=ax, markersize=10)
            
            plt.show()
    
    def calc_horizontal_segment_stats(self):
        point_trimmed_list = list(
                zip([len(point) 
                for point
                in self.horiz_segment_intersected_points], 
                self.horizontal_segments['masked_length'])
                )
        
        self.horizontal_segments['p10_masked'] = (
                [x[0]/x[1] if x[1] > 0 else np.nan 
                 for x in point_trimmed_list]
                )
       
        print('Horizontal segment stats calculated')
        
    def calc_vertical_segment_stats(self):
        point_trimmed_list = list(
                zip([len(point) 
                for point 
                in self.vert_segment_intersected_points], 
                self.vertical_segments['masked_length'])
                )
        
        self.vertical_segments['p10_masked'] = (
                [x[0]/x[1] if x[1] > 0 else np.nan 
                 for x in point_trimmed_list]
                )
        
        print('Vertical segment stats calculated')
        
    def calc_segment_stats(self):
        if self.limit_direction_to != 'vertical':
            self.calc_horizontal_segment_stats()
            
        if self.limit_direction_to != 'horizontal':
            self.calc_vertical_segment_stats()
            
    def write_segment_tables(self):
        if self.limit_direction_to != 'vertical':
            self.horizontal_segments.to_csv('horizontal_segments.csv')
            
        if self.limit_direction_to != 'horizontal':
            self.vertical_segments.to_csv('vertical_segments.csv')
            
    def make_windows(self):
        vert_limits = list(self.traces.total_bounds[i] for i in [1,3])
        horiz_limits = list(self.traces.total_bounds[i] for i in [0,2])
        
        x_coords = np.arange(
                min(horiz_limits) + self.window_step_increment_m/2, 
                max(horiz_limits), self.window_step_increment_m
                )
        
        y_coords = np.arange(
                min(vert_limits) + self.window_step_increment_m/2, 
                max(vert_limits), self.window_step_increment_m
                )
       
        x,y,w = np.meshgrid(x_coords, y_coords, self.window_width_m)
        x_array = np.concatenate(np.concatenate(x))
        y_array = np.concatenate(np.concatenate(y))
        w_array = np.concatenate(np.concatenate(w))
        
        polygons = ([make_polygon_from_tuple(x,y,w) 
                     for (x,y,w) in zip(x_array, y_array, w_array)]
                   )
        
        names = ['window_' + str(i) for i in np.arange(0,len(polygons))+1]
                
        self.windows = gpd.GeoDataFrame({
                    'name': names,
                    'y_coord': x_array,
                    'x_coord': y_array},
                    geometry = polygons
                    )
            
        self.windows['orig_width'] = (
                self.windows.bounds.iloc[:,2] 
                - self.windows.bounds.iloc[:,0]
                )
        
        self.windows['orig_height'] = (
                self.windows.bounds.iloc[:,3] 
                - self.windows.bounds.iloc[:,1]
                )
        
        self.windows['orig_area'] = self.windows.area
        
        print('Windows generated')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            
            for index,row in self.windows.iterrows():
                if row.geometry.is_empty:
                    []
                elif isinstance(row.geometry,geometry.multipolygon.MultiPolygon):
                    for polygon in row.geometry:
                        plt.plot(*polygon.exterior.xy)
                else:
                    ax.plot(*row.geometry.exterior.xy)
                
            plt.show()
            
    def mask_windows(self):
        self.windows_orig = self.windows
        
        for mask in self.masks: 
            for (i,window) in enumerate(self.windows.geometry):
                self.windows.geometry[i] = self.windows.geometry[i].difference(mask)
        
        self.windows['masked_geom'] = self.windows.geometry
        
        self.windows['masked_width'] = (
                self.windows.bounds.iloc[:,2] 
                - self.windows.bounds.iloc[:,0]
                )
        
        self.windows['masked_height'] = (
                self.windows.bounds.iloc[:,3] 
                - self.windows.bounds.iloc[:,1]
                )
        
        self.windows['masked_area'] = self.windows.area
        self.windows['masked_length'] = self.windows.length
        
        print('Masking windows (saved & overwritten)')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            
            for mask in self.masks:
                ax.plot(*mask.exterior.xy, color = 'k')
            
            for index,row in self.windows.iterrows():
                if row.geometry.is_empty:
                    []
                elif isinstance(row.geometry,geometry.multipolygon.MultiPolygon):
                    for polygon in row.geometry:
                        plt.plot(*polygon.exterior.xy)
                else:
                    ax.plot(*row.geometry.exterior.xy)
            
            plt.show()
        
    def intersect_windows(self):
        self.windows_intersections = [
                self.traces.intersection(other = window) if
                ~window.is_empty else Polygon()
                for window
                in self.windows.geometry
                ]
        
        self.windows_intersected_traces = [
                intersection[np.invert(intersection.is_empty)] 
                for intersection
                in self.windows_intersections
                ]
        
        print('Windows and traces intersected')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            
            for index,row in self.windows.iterrows():
                if row.geometry.is_empty:
                    []
                elif isinstance(row.geometry,geometry.multipolygon.MultiPolygon):
                    for polygon in row.geometry:
                        ax.plot(*polygon.exterior.xy)
                else:
                    ax.plot(*row.geometry.exterior.xy)
            
            [x.plot(ax=ax, color = 'r') 
            for x 
            in self.windows_intersected_traces
            if ~x.is_empty.all() 
            ]
            
            plt.show()
        
    def calc_window_stats(self):
        trace_count_list = list(
                zip([traces.count()
                for traces
                in self.windows_intersected_traces], 
                self.windows['masked_area'])
                )
        
        self.windows['p20_masked'] = (
                [x[0]/x[1] if x[1] > 0 else np.nan 
                 for x in trace_count_list]
                )
       
        trace_length_list = list(
                zip([traces.length.sum()
                for traces
                in self.windows_intersected_traces], 
                self.windows['masked_area'])
                )
        
        self.windows['p21_masked'] = (
                [x[0]/x[1] if x[1] > 0 else np.nan 
                 for x in trace_length_list]
                )
        
        print('Window stats calculated')
        
    def write_window_table(self):
        self.windows.to_csv('windows.csv')
