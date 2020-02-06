def calc_row_idx(k, n):
    return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))/2
    
def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

# -*- coding: utf-8 -*-
class Variogram(object):
    """A class to analyze and model experimental variograms of a 
    GeostatsDataFrame object"""
    
    xlag_dist = 50
    ylag_dist = 50
    lag_dist_tol = 2
    n_lags = 50
    azimuth_cw_ns = 90
    azi_tol = 1
    dlag = 10
    bandwidth = 10
    epsilon = 1.0e-20
    
    def __init__(self, geostats_df):
        self.gs_df = geostats_df

    def convert_azimuth(self):
        """ Mathematical azimuth is measured counterclockwise from EW and
        not clockwise from NS as the conventional azimuth """
        self.azimuth_ccw_ew = (90.0 - self.azimuth_cw_ns) * math.pi / 180.0
         
        if self.azi_tol <= 0.0:
            self.azi_tol_ccw = math.cos(45.0 * math.pi / 180.0)
        else:
            self.azi_tol_ccw = math.cos(self.azi_tol * math.pi / 180.0)

    def condensed_to_square(self, k):
        i = calc_row_idx(k, len(self.gs_df))
        j = calc_col_idx(k, i, len(self.gs_df))
        return i, j

    def calculate_lags(self):
        h = ssd.pdist(self.gs_df[['x','y']])
        h_filt = h.copy()
        
        max_dist = (
                (float(self.n_lags) + 0.5 - self.epsilon) 
                * max(self.xlag_dist, self.ylag_dist)
                )
        
        h_filt[h > max_dist] = np.nan
        h_filt[h < self.epsilon] = np.nan
        
        self.lags = pd.DataFrame(h_filt, columns = ('lag_dist',))

    def map_values_to_lags(self, k):
        pairs = pd.DataFrame(
                [self.condensed_to_square(i) for i in range(1,len(self.lags))],
                columns = ('pt_1', 'pt_2')
                )
        
        self.lags = pd.concat([self.lags.reset_index(drop=True), pairs], axis=1)
        
        df_c = pd.concat([df_a
       

def variogram_loop(x, y, vr, xlag, xltol, nlag, azm, atol, bandwh):
    """Calculate the variogram by looping over combinatorial of data pairs.
    :param x: x values
    :param y: y values
    :param vr: property values
    :param xlag: lag distance
    :param xltol: lag distance tolerance
    :param nlag: number of lags to calculate
    :param azm: azimuth
    :param atol: azimuth tolerance
    :param bandwh: horizontal bandwidth / maximum distance offset orthogonal to
                   azimuth
    :return: TODO
    """
    # Allocate the needed memory
    
    nvarg = 1
    mxdlv = nlag + 2  # in gamv the npp etc. arrays go to nlag + 2
    dis = np.zeros(mxdlv)
    lag = np.zeros(mxdlv)  # TODO: not used
    vario = np.zeros(mxdlv)
    hm = np.zeros(mxdlv)
    tm = np.zeros(mxdlv)
    hv = np.zeros(mxdlv)  # TODO: not used
    npp = np.zeros(mxdlv)
    ivtail = np.zeros(nvarg + 2)
    ivhead = np.zeros(nvarg + 2)
    ivtype = np.ones(nvarg + 2)
    ivtail[0] = 0
    ivhead[0] = 0
    ivtype[0] = 0
    nsiz = nlag + 2  # TODO: not used
    nd = len(x)    

    
                # Determine which lag this is and skip if outside the defined
                # distance tolerance
                if h <= EPSLON:
                    lagbeg = 0
                    lagend = 0
                else:
                    lagbeg = -1
                    lagend = -1
                    for ilag in range(1, nlag + 1):
                        # reduced to -1
                        if (
                            (xlag * float(ilag - 1) - xltol)
                            <= h
                            <= (xlag * float(ilag - 1) + xltol)
                        ):
                            if lagbeg < 0:
                                lagbeg = ilag
                            lagend = ilag
                if lagend >= 0:
                    # Definition of the direction corresponding to the current
                    # pair. All directions are considered (overlapping of
                    # direction tolerance cones is allowed)

                    # Check for an acceptable azimuth angle
                    dxy = np.sqrt(max((dxs + dys), 0.0))
                    if dxy < EPSLON:
                        dcazm = 1.0
                    else:
                        dcazm = (dx * math.cos(azimuth_ccw_ew) + dy * math.sin(azimuth_ccw_ew)) / dxy

                    # Check the horizontal bandwidth criteria (maximum deviation
                    # perpendicular to the specified direction azimuth)
                    band = math.cos(azimuth_ccw_ew) * dy - math.sin(azimuth_ccw_ew) * dx

                    # Apply all the previous checks at once to avoid a lot of
                    # nested if statements
                    if (abs(dcazm) >= csatol) and (abs(band) <= bandwh):
                        # Check whether or not an omni-directional variogram is
                        # being computed
                        omni = False
                        if atol >= 90.0:
                            omni = True

                        # For this variogram, sort out which is the tail and
                        # the head value
                        if dcazm >= 0.0:
                            vrh = vr[i]
                            vrt = vr[j]
                            if omni:
                                vrtpr = vr[i]
                                vrhpr = vr[j]
                        else:
                            vrh = vr[j]
                            vrt = vr[i]
                            if omni:
                                vrtpr = vr[j]
                                vrhpr = vr[i]

                        # Reject this pair on the basis of missing values

                        # Data was trimmed at the beginning

                        # The Semivariogram (all other types of measures are
                        # removed for now)
                        for il in range(lagbeg, lagend + 1):
                            npp[il] = npp[il] + 1
                            dis[il] = dis[il] + h
                            tm[il] = tm[il] + vrt
                            hm[il] = hm[il] + vrh
                            vario[il] = vario[il] + ((vrh - vrt) * (vrh - vrt))
                            if omni:
                                npp[il] = npp[il] + 1.0
                                dis[il] = dis[il] + h
                                tm[il] = tm[il] + vrtpr
                                hm[il] = hm[il] + vrhpr
                                vario[il] = vario[il] + (
                                    (vrhpr - vrtpr) * (vrhpr - vrtpr)
                                )

    # Get average values for gam, hm, tm, hv, and tv, then compute the correct
    # "variogram" measure
    for il in range(0, nlag + 2):
        i = il
        if npp[i] > 0:
            rnum = npp[i]
            dis[i] = dis[i] / rnum
            vario[i] = vario[i] / rnum
            hm[i] = hm[i] / rnum
            tm[i] = tm[i] / rnum

    return dis, vario, npp

    
    def varmap2d(self, val_col):
        
        pd.Data
        
    
    
    
    
    
    
    
    
    
   
    for index, row in ij_tuple.iterrows():
      y_dis = self.gs_df['y'][row.j] - self.gs_df['y'][i]
      x_dis = self.gs_df['x'][j] - self.gs_df['x'][i]
      iyl = lags + int(ydis/dlag)
      ixl = lags + int(ydis/dlag)
      df_out = pd.DataFrame({'y':y_dis, 'x':x_dis, 'iy':iyl, 'ix':ixl})
      
    ij_tuple.['i']
      
    def ydis(i,j):
        
    geostats_df
    
     for i in range(0,nrow):     
            for j in range(0,nrow): 
                ydis = y[j] - y[i]
                iyl = lags + int(ydis/dlag)
                if iyl < 0 or iyl > lags*2: # acocunting for 0,...,n-1 array indexing
                    continue
                xdis = x[j] - x[i]
                ixl = lags + int(xdis/dlag)
                if ixl < 0 or ixl > lags*2: # acocunting for 0,...,n-1 array indexing
                    continue
                    
                # We have an acceptable pair, therefore accumulate all the statistics
                # that are required for the variogram:
                npp[iyl,ixl] = npp[iyl,ixl] + 1 # our ndarrays read from the base to top, so we flip
                tm[iyl,ixl] = tm[iyl,ixl] + val[i]
                hm[iyl,ixl] = hm[iyl,ixl] + val[j]
                tv[iyl,ixl] = tm[iyl,ixl] + val[i]*val[i]
                hv[iyl,ixl] = hm[iyl,ixl] + val[j]*val[j]
                gam[iyl,ixl] = gam[iyl,ixl] + ((val[i]-val[j])*(val[i]-val[j]))
    
    for i,j in range(0,10):
        print(i)
        print(j)
    
    
    xmax = ((float(nx)+0.5)*lagdist); xmin = -1*xmax; 
    ymax = ((float(ny)+0.5)*lagdist); ymin = -1*ymax;
    
    
    def variogram_map(self, column, lags = 10, dlag = 1):
        """Simplified reimplementation of Deutsch and Journel (1998) variogram 
        map with a standardized sill and GeostatsDataFrame object"""
        
        nrow = len(self.gs_df)
        
        # initialize arrays
        zeroes = np.zeros((lags*2+1,lags*2+1))
        npp, gam, nppf, gamf, hm, tm, hv, tv = (
                zeroes, zeroes, zeroes, zeroes,zeroes,zeroes,zeroes,zeroes
                )
    
        x = np.array(self.gs_df['x'])
        y = np.array(self.gs_df['y'])
        val = np.array(self.gs_df[column])
 
       
    
        # Get average values for gam, hm, tm, hv, and tv, then compute
        # the correct "variogram" measure:
        for iy in range(0,lags*2+1): 
            for ix in range(0,lags*2+1): 
                if npp[iy,ix] <= 1:
                    gam[iy,ix] = -999.
                    hm[iy,ix]  = -999.
                    tm[iy,ix]  = -999.
                    hv[iy,ix]  = -999.
                    tv[iy,ix]  = -999.
                else:
                    rnum = npp[iy,ix]
                    gam[iy,ix] = gam[iy,ix] / (2*rnum) # semivariogram
                    hm[iy,ix] = hm[iy,ix] / rnum
                    tm[iy,ix] = tm[iy,ix] / rnum
                    hv[iy,ix] = hv[iy,ix] / rnum - hm[iy,ix]*hm[iy,ix]
                    tv[iy,ix] = tv[iy,ix] / rnum - tm[iy,ix]*tm[iy,ix]
                    
        # Standardize
        gamf[iy,ix] = gamf[iy,ix]/df[column].std()**2
    
        for iy in range(0,lags*2+1): 
            for ix in range(0,lags*2+1):             
                gamf[iy,ix] = gam[lags*2-iy,ix]
                nppf[iy,ix] = npp[lags*2-iy,ix]
                
        self.vmap = gamf
        self.npmap = nppf 
    
    def variogram_map(self, column, lags = 10, dlag = 1):
        plt.subplot(121)
        GSLIB.pixelplt_st(self.vmap,-575,575,-575,575,50.0,0,1.6,'Nscore Porosity Variogram Map','X(m)','Y(m)','Nscore Variogram',cmap)

    def variogram_plot(self, step = 100):
         xx, yy = np.meshgrid(
                 np.linspace(self.gs_df.x.min(), self.gs_df.x.max(), self.gamf.shape[0]), 
                 np.linspace(self.gs_df.y.max(), self.gs_df.y.min(), self.gamf.shape[0])
                 )
         
         cs = plt.contourf(xx, yy, self.gamf,)
        xx,
        yy,
        array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        levels=np.linspace(vmin, vmax, 100),
    )

def pixelplt_st(
    array,
    xmin,
    xmax,
    ymin,
    ymax,
    step,
    vmin,
    vmax,
    title,
    xlabel,
    ylabel,
    vlabel,
    cmap,
):
    """Pixel plot, reimplementation in Python of GSLIB pixelplt with Matplotlib
    methods (version for subplots).

    :param array: ndarray
    :param xmin: x axis minimum
    :param xmax: x axis maximum
    :param ymin: y axis minimum
    :param ymax: y axis maximum
    :param step: step
    :param vmin: TODO
    :param vmax: TODO
    :param title: title
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param vlabel: TODO
    :param cmap: colormap
    :return: QuadContourSet
    """
   

    # Use dummy since scatter plot controls legend min and max appropriately
    # and contour does not!
    x = []
    y = []
    v = []

    cs = plt.contourf(
        xx,
        yy,
        array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        levels=np.linspace(vmin, vmax, 100),
    )
    im = plt.scatter(
        x,
        y,
        s=None,
        c=v,
        marker=None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        linewidths=0.8,
        verts=None,
        edgecolors="black",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.clim(vmin, vmax)
    cbar = plt.colorbar(im, orientation="vertical")
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs
