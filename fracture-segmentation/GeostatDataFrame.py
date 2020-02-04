# -*- coding: utf-8 -*-
class GeostatsDataFrame(object):
    """A class to load an transform a table of xy + feature values into
    a GSLIB compliant dataframe for variogram modelling and simulation"""
    
    def __init__(self):
        []
        
    def load_csv(self, filepath):
        self.input = pd.read_csv(filepath)
        
    