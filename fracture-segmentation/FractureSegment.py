class FractureSegment(object):
    """A class to contain fracture segmentation results"""
    
    sig_spatial = 5
    sig_color = 3
    sig_threshold = 0.5
    denoise_window_px = 5
    gap_fill_px = 3
    show_figures = False
    save_figures = False
    canny_method = 'horizontal'
    min_large_edge_px = 50
    phough_min_line_length_px = 50
    phough_line_gap_px = 10
    phough_accumulator_threshold = 100
    
    def __init__(self, filepath):
        self.img = io.imread(filepath, as_gray = True)
    
    def list_params(self):
        """ Print a list of object parameters """
        print('sig_spatial: ' + str(self.sig_spatial))
        print('sig_color: ' + str(self.sig_color))
        print('sig_threshold: ' + str(self.sig_threshold))
        print('denoise_window_px: ' + str(self.denoise_window_px))
        print('gap_fill_px: ' + str(self.gap_fill_px)) 
        print('show_figures: ' + str(self.show_figures))
        print('save_figures: ' + str(self.save_figures))
        print('canny_method: ' + str(self.canny_method))
        print('min_large_edge_px: ' + str(self.min_large_edge_px))
        print('phough_min_line_length_px: ' + str(self.phough_min_line_length_px))
        print('phough_line_gap_px: ' + str(self.phough_line_gap_px))
        print('phough_accumulator_threshold: ' + str(self.phough_accumulator_threshold))
        
    def show_img(self):
        """ Show image using io.imshow and matplotlib """
        io.imshow(self.img)
        plt.show()
    
    def denoise(self):
        """ Run a bilateral denoise on the raw image """
        print('Denoising Image')
        
        self.img_denoised = denoise_bilateral(
                self.img, sigma_spatial = self.sig_spatial, 
                sigma_color = self.sig_color, multichannel=False,
                win_size = self.pixel_window)
        
        if self.show_figures:
            io.imshow(self.img_denoised)
            plt.show()
        
        if self.save_figures:
            io.imsave('./output/img_denoised.png',util.img_as_float(self.img_denoised))
    
    def detect_edges(self):
        """ Run one of several modified Canny edge detectors on the denoised
            image.    
        """
        if self.canny_method == 'horizontal':
            print('Running Horizontal One-Way Gradient Canny Detector')
            self.img_edges = canny_horiz(self.img_denoised, self.sig_threshold)
        else: 
            print('Running Standard Canny Detector')
            self.img_edges = canny_std(self.img_denoised, self.sig_threshold)
        
        if self.show_figures:
            io.imshow(self.img_edges)
            plt.show()
        
        if self.save_figures:
            io.imsave('./output/img_edges.png',util.img_as_ubyte(self.img_edges))
            
    def close_gaps(self):
        """ Close small holes with binary closing to within x pixels """
        
        print('Closing binary gaps')
        
        self.img_closededges = binary_closing(self.img_edges, square(self.gap_fill_px))
        
        if self.show_figures:
            io.imshow(self.img_closededges)
            plt.show()
        
        if self.save_figures:
            io.imsave('./output/img_closededges.png',util.img_as_ubyte(self.img_closededges))
    
    def label_edges(self):
        """ Label connected edges/components using skimage wrapper """
        
        print('Labelling connected edges')
        
        self.img_labelled = measure.label(self.img_closededges, 
                                          connectivity=2, background=0)
        
        print(str(len(np.unique(sample.img_labelled))-1) + ' components identified')
             
        if self.show_figures:
            io.imshow(self.img_labelled)
            plt.show()
        
    def count_edges(self):
        """ Get a unique count of edges, omitting zero values  """       
        unique, counts = np.unique(sample.img_labelled, return_counts=True)
        self.edge_dict = dict(zip(unique, counts))
        self.edge_dict.pop(0)
        
        edge_cov = sum(self.edge_dict.values()) / self.img_labelled.size * 100
        
        print(str(edge_cov) + '% edge coverage')
        
    def find_large_edges(self):
        """ Label connected edges/components using skimage wrapper """ 
        self.large_edge_dict = {k: v for k, v 
                                in self.edge_dict.items()
                                if v >= self.min_large_edge_px}
        
        large_edge_cov = len(self.large_edge_dict) / len(self.edge_dict) * 100
        
        print(str(large_edge_cov) + '% large edges')
        
        large_edge_bool = np.isin(self.img_labelled, list(self.large_edge_dict.keys()))
        
        self.img_large_edges = self.img_labelled.copy()
        self.img_large_edges[np.invert(large_edge_bool)] = 0

        if self.show_figures:
            io.imshow(self.img_large_edges)
            plt.show()

        if self.save_figures:
            io.imsave('./output/img_large_edges.png',util.img_as_ubyte(self.img_large_edges > 0))
            
    def run_phough_transform(self):
        """ Run the Probabilistic Hough Transform """
        print('Running Probabilistic Hough Transform')
        
        self.lines = probabilistic_hough_line(
                self.img_large_edges,    
                line_length=self.phough_min_line_length_px,
                line_gap=self.phough_line_gap_px,
                threshold = self.phough_accumulator_threshold)
        
        if self.show_figures:
            fig, ax = plt.subplots(1, 1)
            io.imshow(self.img_large_edges * 0)
            for line in self.lines:
                p0, p1 = line
                ax.plot((p0[0], p1[0]), (p0[1], p1[1]))
            ax.set_xlim((0, self.img_large_edges.shape[1]))
            ax.set_ylim((self.img_large_edges.shape[0], 0))
            plt.show()
    

