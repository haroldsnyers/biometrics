# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:44:22 2018

@author: Utkarsh
"""

import cv2
import numpy as np
from skimage.morphology import erosion, square


def removeSpuriousMinutiae(minutiaeList, dist_radius, minutiaeTerm, mask):
    
    # %% first, merge endings within dist_radius
    minutiaeTerm = minutiaeTerm * 0;
    SpuriousMin = [];
    numPoints = len(minutiaeList);
    D = np.zeros((numPoints, numPoints))
    for i in range(1,numPoints):
        for j in range(0, i):
            (X1,Y1) = minutiaeList[i]['centroid']
            (X2,Y2) = minutiaeList[j]['centroid']
            
            dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2);
            D[i][j] = dist
            if(dist < dist_radius):
                SpuriousMin.append(i)
                SpuriousMin.append(j)
                
    SpuriousMin = np.unique(SpuriousMin)
    for i in range(0,numPoints):
        if(not i in SpuriousMin):
            (X,Y) = np.int16(minutiaeList[i]['centroid']);
            minutiaeTerm[X,Y] = 1;

    # %% second, remove endings near the border of the mask
    
    minutiaeTerm = np.uint8(minutiaeTerm);
	# make background 1 pixel layer border
    border_width = 1
    mask_with_border = cv2.copyMakeBorder(np.uint8(mask), 
                                       border_width, border_width, border_width, border_width,
                                       cv2.BORDER_CONSTANT, 0)
    mask = erosion(mask_with_border, square(15))
    mask = mask[border_width:-border_width, border_width: -border_width]
    minutiaeTerm = cv2.bitwise_and(minutiaeTerm, minutiaeTerm, mask=np.uint8(mask))

    return(minutiaeTerm)