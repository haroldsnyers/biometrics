# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:50:30 2016

@author: utkarsh
"""
from .ridge_segment import ridge_segment
from .ridge_orient import ridge_orient
from .ridge_freq import ridge_freq
from .ridge_filter import ridge_filter

def image_enhance(img):

    # normalise the image and find a ROI
    blksze = 16
    thresh = 0.1
    normim,mask = ridge_segment(img,blksze,thresh)

    # find orientation of every pixel using gradients (smoothed)
    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma);

    #find the overall frequency of ridges
    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freqim,medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,maxWaveLength);

    # create gabor filter and do the actual filtering
    freq = medfreq*mask
    kx = 0.65;ky = 0.65
    enhim = ridge_filter(normim, orientim, freq, kx, ky)


    #th, bin_im = cv2.threshold(np.uint8(newim),0,255,cv2.THRESH_BINARY);
#    return(newim < -3)
    return(enhim, mask, orientim, freqim)
