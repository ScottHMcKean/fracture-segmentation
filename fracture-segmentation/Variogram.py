# -*- coding: utf-8 -*-
class Variogram(object):
    """A class to analyze and model experimental variograms of a 
    GeostatsDataFrame object"""
    
    def __init__(self, geostats_df):
        self.gs_df = geostats_df
    
    
    
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
