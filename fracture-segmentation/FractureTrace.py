class FractureTrace(object):
    """A class to contain the results of fracture trace analysis from 
    a vector file (shp,dxf,etc., or FractureImage object results"""
    
    show_figures = False
    save_figures = False
    segment_width_m = 1
    segment_step_increment_m = 0.1
    scanline_distance_m = 0.1
    scale_m_px = 1
    
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
        print('Overwritting original traces with scaled traces')
        
        if hasattr(self, 'masks'):
            self.masks = self.masks.affine_transform(matrix)
            print('Overwritting original masks with scaled traces')
        
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
        
        print('Saving original traces as traces_orig')
        print('Overwritting original traces with masked traces')
        
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
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            self.horizontal_scanlines.plot(color = 'r', ax=ax)
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
        names = ['scan_v' + str(i) for i in np.arange(0,len(lines))+1]
            
        self.vertical_scanlines = gpd.GeoDataFrame({
                'name': names,
                'x_coord': horiz_splits},
                geometry =  gpd.GeoSeries(map(LineString, lines))
                )
        
        self.vertical_scanlines['orig_length'] = self.vertical_scanlines.length
        self.vertical_scanlines['orig_geom'] = self.vertical_scanlines['geometry']
        
        print('Vertical scanlines generated')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            self.vertical_scanlines.plot(color = 'r', ax=ax)
            plt.show()
            
    def make_scanlines(self):
        self.make_vertical_scanlines()
        self.make_horizontal_scanlines()
        
    def mask_horizontal_scanlines(self):
        self.horizontal_scanlines_orig = self.horizontal_scanlines
        
        for mask in self.masks: 
            for (i,line) in enumerate(self.horizontal_scanlines.geometry):
                self.horizontal_scanlines.geometry[i] = self.horizontal_scanlines.geometry[i].difference(mask)
        
        self.horizontal_scanlines['masked_geom'] = self.horizontal_scanlines.geometry
        self.horizontal_scanlines['masked_length'] = self.horizontal_scanlines.length
        
        print('Saving original horizontal scanlines as horizontal_scanlines_orig')
        print('Overwritting original horizontal scanlines with masked scanlines')
        
        if self.show_figures:
                fig, ax = plt.subplots(1, 1)
                self.horizontal_scanlines_orig[~self.horizontal_scanlines_orig.is_empty].plot(color = 'k', ax=ax, alpha = 0.5)
                self.horizontal_scanlines[~self.horizontal_scanlines.is_empty].plot(color = 'r', ax=ax)
                for mask in self.masks:
                    ax.plot(*mask.exterior.xy, color = 'b')
                plt.show()
                
    def mask_vertical_scanlines(self):
        self.vertical_scanlines_orig = self.vertical_scanlines
        
        for mask in self.masks: 
            for (i,line) in enumerate(self.vertical_scanlines.geometry):
                self.vertical_scanlines.geometry[i] = self.vertical_scanlines.geometry[i].difference(mask)
        
        self.vertical_scanlines['masked_geom'] = self.vertical_scanlines.geometry
        self.vertical_scanlines['masked_length'] = self.vertical_scanlines.length
        
        print('Saving original vertical scanlines as vertical_scanlines_orig')
        print('Overwritting original vertical scanlines with masked scanlines')
        
        if self.show_figures:
                fig, ax = plt.subplots(1, 1)
                self.vertical_scanlines_orig[~self.vertical_scanlines_orig.is_empty].plot(color = 'k', ax=ax, alpha = 0.5)
                self.vertical_scanlines[~self.vertical_scanlines.is_empty].plot(color = 'r', ax=ax)
                for mask in self.masks:
                    ax.plot(*mask.exterior.xy, color = 'b')
                plt.show()
  
    def mask_scanlines(self):
        self.mask_vertical_scanlines()
        self.mask_horizontal_scanlines()

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
        self.horizontal_scanlines.geometry = [
                self.total_convex_hull.intersection(x) 
                    for x 
                    in self.horizontal_scanlines.geometry]
    
        self.horizontal_scanlines['hull_trimmed'] = self.horizontal_scanlines.geometry
        self.horizontal_scanlines['trimmed_length'] = self.horizontal_scanlines.length
        
        self.vertical_scanlines.geometry = [
                self.total_convex_hull.intersection(x) 
                    for x 
                    in self.vertical_scanlines.geometry]
            
        self.vertical_scanlines['hull_trimmed'] = self.vertical_scanlines.geometry
        self.vertical_scanlines['trimmed_length'] = self.vertical_scanlines.length
        
        print('Scanlines trimmed')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            ax.plot(*self.total_convex_hull.exterior.xy, color = 'b')
            self.horizontal_scanlines[~self.horizontal_scanlines.is_empty].plot(color = 'r', ax=ax)
            self.vertical_scanlines[~self.vertical_scanlines.is_empty].plot(color = 'r', ax=ax)
            plt.show()
                
    def intersect_horizontal_scanlines_traces(self):
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
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax, alpha=0.5)
            ax.plot(*self.total_convex_hull.exterior.xy, color = 'k', alpha=0.5)
            self.horizontal_scanlines[~self.horizontal_scanlines.is_empty].plot(color = 'k', ax=ax, alpha = 0.5)
            
            traces_series = convert_geo_list_to_geoseries(
                    self.horiz_scanline_intersected_traces
                    )
            
            points_series = convert_geo_list_to_geoseries(
                    self.horiz_scanline_intersected_points
                    )
            
            traces_series.plot(color = 'r', ax=ax, markersize=5)
            points_series.plot(color = 'r', ax=ax, markersize=5)
            plt.show()
    
    def intersect_vertical_scanlines_traces(self):
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
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax, alpha=0.5)
            ax.plot(*self.total_convex_hull.exterior.xy, color = 'k', alpha=0.5)
            self.vertical_scanlines[~self.vertical_scanlines.is_empty].plot(color = 'k', ax=ax, alpha = 0.5)
            
            traces_series = convert_geo_list_to_geoseries(
                    self.vert_scanline_intersected_traces
                    )
            
            points_series = convert_geo_list_to_geoseries(
                    self.vert_scanline_intersected_points
                    )
            
            traces_series.plot(color = 'b', ax=ax, markersize=5)
            points_series.plot(color = 'b', ax=ax, markersize=5)
            plt.show()
    
    def intersect_scanlines_traces(self):
        self.intersect_horizontal_scanlines_traces()
        self.intersect_vertical_scanlines_traces()
    
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
        
        self.horizontal_scanlines['p10_frac'] = [x[0]/x[1] if x[1] > 0 else np.nan for x in point_frac_list]
        
        point_trimmed_list = list(
                zip([len(point) 
                for point
                in self.horiz_scanline_intersected_points], 
                self.horizontal_scanlines['trimmed_length'])
                )
        
        self.horizontal_scanlines['p10_trimmed'] = [x[0]/x[1] if x[1] > 0 else np.nan for x in point_trimmed_list]
       
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
        
        self.vertical_scanlines['p10_frac'] = [x[0]/x[1] if x[1] > 0 else np.nan for x in point_frac_list]
        
        point_trimmed_list = list(
                zip([len(point) 
                for point 
                in self.vert_scanline_intersected_points], 
                self.vertical_scanlines['trimmed_length'])
                )
        
        self.vertical_scanlines['p10_trimmed'] = [x[0]/x[1] if x[1] > 0 else np.nan for x in point_trimmed_list]
        
        print('Vertical scanline stats calculated')
        
    def calc_scanline_stats(self):
        self.calc_horizontal_scanline_stats()
        self.calc_vertical_scanline_stats()
      
    def make_scanline_segments(self):
        
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
        
    def mask_horizontal_segments(self):
        self.horizontal_segments_orig = self.horizontal_segments
        
        for mask in self.masks: 
            for (i,line) in enumerate(self.horizontal_segments.geometry):
                self.horizontal_segments.geometry[i] = self.horizontal_segments.geometry[i].difference(mask)
        
        self.horizontal_segments['masked_geom'] = self.horizontal_segments.geometry
        self.horizontal_segments['masked_length'] = self.horizontal_segments.length
        
        print('Saving original horizontal segments as horizontal_segments_orig')
        print('Overwritting original horizontal segments with masked segments')
        
        if self.show_figures:
                fig, ax = plt.subplots(1, 1)
                self.horizontal_segments_orig[~self.horizontal_segments_orig.is_empty].plot(color = 'k', ax=ax, alpha = 0.5)
                self.horizontal_segments[~self.horizontal_segments.is_empty].plot(color = 'r', ax=ax)
                for mask in self.masks:
                    ax.plot(*mask.exterior.xy, color = 'b')
                plt.show()
                
    def mask_vertical_segments(self):
        self.vertical_segments_orig = self.vertical_segments
        
        for mask in self.masks: 
            for (i,line) in enumerate(self.vertical_segments.geometry):
                self.vertical_segments.geometry[i] = self.vertical_segments.geometry[i].difference(mask)
        
        self.vertical_segments['masked_geom'] = self.vertical_segments.geometry
        self.vertical_segments['masked_length'] = self.vertical_segments.length
        
        print('Saving original vertical segments as vertical_segments_orig')
        print('Overwritting original vertical segments with masked segments')
        
        if self.show_figures:
                fig, ax = plt.subplots(1, 1)
                self.vertical_segments_orig[~self.vertical_segments_orig.is_empty].plot(color = 'k', ax=ax, alpha = 0.5)
                self.vertical_segments[~self.vertical_segments.is_empty].plot(color = 'r', ax=ax)
                for mask in self.masks:
                    ax.plot(*mask.exterior.xy, color = 'b')
                plt.show()
  
    def mask_segments(self):
        self.mask_vertical_segments()
        self.mask_horizontal_segments()
        
    def intersect_segments_hull(self):
        self.horizontal_segments.geometry = [
                self.total_convex_hull.intersection(x) 
                    for x 
                    in self.horizontal_segments.geometry]
    
        self.horizontal_segments['hull_trimmed'] = self.horizontal_segments.geometry
        self.horizontal_segments['trimmed_length'] = self.horizontal_segments.length
        
        self.vertical_segments.geometry = [
                self.total_convex_hull.intersection(x) 
                    for x 
                    in self.vertical_segments.geometry]
            
        self.vertical_segments['hull_trimmed'] = self.vertical_segments.geometry
        self.vertical_segments['trimmed_length'] = self.vertical_segments.length
        
        print('Segments trimmed')
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax)
            ax.plot(*self.total_convex_hull.exterior.xy, color = 'b')
            self.horizontal_segments[~self.horizontal_segments.is_empty].plot(color = 'r', ax=ax)
            self.vertical_segments[~self.vertical_segments.is_empty].plot(color = 'r', ax=ax)
            plt.show()
                
    def intersect_horizontal_segments_traces(self):
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
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax, alpha=0.5)
            ax.plot(*self.total_convex_hull.exterior.xy, color = 'k', alpha=0.5)
            self.horizontal_segments[~self.horizontal_segments.is_empty].plot(color = 'k', ax=ax, alpha = 0.5)
            
            traces_series = convert_geo_list_to_geoseries(
                    self.horiz_segment_intersected_traces
                    )
            
            points_series = convert_geo_list_to_geoseries(
                    self.horiz_segment_intersected_points
                    )
            
            traces_series.plot(color = 'r', ax=ax, markersize=5)
            points_series.plot(color = 'r', ax=ax, markersize=5)
            plt.show()
    
    def intersect_vertical_segments_traces(self):
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
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            self.traces.plot(color = 'k', ax=ax, alpha=0.5)
            ax.plot(*self.total_convex_hull.exterior.xy, color = 'k', alpha=0.5)
            self.vertical_segments[~self.vertical_segments.is_empty].plot(color = 'k', ax=ax, alpha = 0.5)
            
            traces_series = convert_geo_list_to_geoseries(
                    self.vert_segment_intersected_traces
                    )
            
            points_series = convert_geo_list_to_geoseries(
                    self.vert_segment_intersected_points
                    )
            
            traces_series.plot(color = 'b', ax=ax, markersize=5)
            points_series.plot(color = 'b', ax=ax, markersize=5)
            plt.show()
    
    def intersect_segments_traces(self):
        self.intersect_horizontal_segments_traces()
        self.intersect_vertical_segments_traces()
    
    def calc_horizontal_segment_stats(self):
        
        self.horizontal_segments['frac_to_frac_length'] = [
                max(points.x) - min(points.x) 
                if len(points) > 0
                else np.nan
                for points
                in self.horiz_segment_intersected_points
                ]
        
        point_frac_list = list(
                zip([len(point) 
                for point
                in self.horiz_segment_intersected_points], 
                self.horizontal_segments['frac_to_frac_length'])
                )
        
        self.horizontal_segments['p10_frac'] = [x[0]/x[1] if x[1] > 0 else np.nan for x in point_frac_list]
        
        point_trimmed_list = list(
                zip([len(point) 
                for point
                in self.horiz_segment_intersected_points], 
                self.horizontal_segments['trimmed_length'])
                )
        
        self.horizontal_segments['p10_trimmed'] = [x[0]/x[1] if x[1] > 0 else np.nan for x in point_trimmed_list]
       
        print('Horizontal segment stats calculated')
        
    def calc_vertical_segment_stats(self):
        
        self.vertical_segments['frac_to_frac_length'] = [
                max(points.y) - min(points.y) 
                if len(points) > 0
                else np.nan
                for points
                in self.vert_segment_intersected_points
                ]
        
        point_frac_list = list(
                zip([len(point) 
                for point 
                in self.vert_segment_intersected_points], 
                self.vertical_segments['frac_to_frac_length'])
                )
        
        self.vertical_segments['p10_frac'] = [x[0]/x[1] if x[1] > 0 else np.nan for x in point_frac_list]
        
        point_trimmed_list = list(
                zip([len(point) 
                for point 
                in self.vert_segment_intersected_points], 
                self.vertical_segments['trimmed_length'])
                )
        
        self.vertical_segments['p10_trimmed'] = [x[0]/x[1] if x[1] > 0 else np.nan for x in point_trimmed_list]
        
        print('Vertical segment stats calculated')
        
    def calc_segment_stats(self):
        self.calc_horizontal_segment_stats()
        self.calc_vertical_segment_stats()