#' Determine bin numbers for ecdf plots and analysis
#' 
#' @description ECDF functions require binning into discrete categories. This
#' complicates our modelling process by turning a continuous distribution into a 
#' discrete categorical series of bins. There are There are several methods for determining
#' this binning: 
#' Sturges, H. (1926) The choice of a class-interval. J. Amer. Statist. Assoc., 21, 65-66.
#' Scott, D.W. (1979) On optimal and data-based histograms. Biometrika, 66, 605-610.
#' Freedman, D. and Diaconis, P. (1981) On this histogram as a density estimator: L2 theory. Zeit. Wahr. ver. Geb., 57, 453-476.
#' Wand, M. P. (1997). Data-based choice of histogram bin width. The American Statistician, 51(1), 59-64.
#' We use three common methods to calculate the deterministic b-value of 
#' the catalogue. These functions are also used in the bootstrap estimation
#' of uncertainty.
#' @param value_vector a vector of continous values
#' @param method a character of methods: 'rice', 'sturges', 'freedman'
#' @return an integer number of bins
#' @export
k_bins <- function(value_vector, method = 'rice') {
  if (method == 'sturges') {
    bins <- log2(length(value_vector)) + 1
  } else if (method == 'freedman') {
    bins <- (max(value_vector) - min(value_vector)) / (IQR(value_vector) / length(value_vector)^(1/3))
  } else if (method == 'rice') {
    bins <- 2 * length(value_vector) ^ (1/3)
  } else {
    warning('No binning method specified, defaulting to Rice (1951)')
    bins <- 2 * length(value_vector) ^ (1/3)
  }
  bins %>% ceiling(.)
}

#' Function to determine the maximum magnitude of a catalog
#' @param centres bin centres
#' @param rev_ecdf reverse ecdf used for gutenberg richter relationship
#' @return double maximum magnitude of distribution
#' @export
mmaxc <- function(centres, rev_ecdf) {
  spl_pred <- smooth.spline(centres,rev_ecdf) %>% predict(.,deriv=2)
  spl_pred$x[which(abs(spl_pred$y) == max(abs(spl_pred$y)))]
}

#' Make an ecdf
#' @param value_vector a vector of continous values
#' @param method a character of methods: 'rice', 'sturges', 'freedman'
#' @return a dataframe with catalog centres and an ecdf, filtered to above the magnitude of completness
#' @export
make_ecdf <- function(value_vector, method = 'rice') {
  num_bins <- k_bins(value_vector, method)
  
  bins <- seq(round(0.9*min(value_vector),1),
              round(1.1*max(value_vector),1),
              length.out = num_bins+1)
  
  ecdf <- cut(value_vector, bins) %>%
    table(value_vector) %>%
    rowSums() %>%
    rev() %>%
    cumsum() %>%
    rev()
  
  centres <- (bins[2:length(bins)]-bins[1:length(bins)-1])/2+bins[1:length(bins)-1]
  
  print(paste0('mmaxc: ', mmaxc(centres, ecdf)))
  
  data.frame(centres = centres, ecdf = ecdf) %>%  filter(centres >= mmaxc(centres, ecdf))
}