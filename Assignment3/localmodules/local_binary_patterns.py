#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Binary Patterns Class

Created on Thu Mar 28 17:50:19 2019

@author: dvdm, copied fromhttps://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
"""

# import the necessary packages
from skimage import feature
import numpy as np
 
class LBP:
    def __init__(self, numPoints, radius, grid_x=8, grid_y=8):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        self.grid_x = grid_x
        self.grid_y = grid_y
 
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist
    
    def describe_list(self, imageArray):
        lbp = []
        for image in imageArray:
            hist = self.describe_LBPH(image)
            lbp.append(hist)
        return np.array(lbp)

        
    def describe_LBPH(self, image, eps=1e-7):
        # calculate the LBP image
        L = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        # calculate the grid geometry
        lbp_height, lbp_width = L.shape
        py = int(np.floor(lbp_height/self.grid_x))
        px = int(np.floor(lbp_width/self.grid_y))

        E = []
        for row in range(0,self.grid_x):
            for col in range(0,self.grid_y):
                C = L[row*py:(row+1)*py,col*px:(col+1)*px]
                (hist, _) = np.histogram(C.ravel(), bins=np.arange(0, self.numPoints + 3), 
                                 range=(0, self.numPoints + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)

                E.extend(hist)
                
        return np.asarray(E)
    
