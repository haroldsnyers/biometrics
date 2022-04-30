# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:12:44 2018

@author: Utkarsh
Code adapted by dvdm
"""

import numpy as np
import cv2

def getTerminationBifurcation(img, mask):
    img = np.uint8((img == 255));
    
    minutiae = cv2.boxFilter(img, -1, (3, 3), 
                                  normalize = False, 
                                  borderType=cv2.BORDER_REFLECT)
    
    minutiaeTerm = cv2.bitwise_and(img, img, mask = np.uint8(minutiae == 2))
    minutiaeBif = cv2.bitwise_and(np.uint8(minutiae == 4),img)
        
    return(minutiaeTerm, minutiaeBif)