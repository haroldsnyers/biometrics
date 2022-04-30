# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:13:36 2018

@author: Utkarsh
Code apdapted by dvdm
"""

import cv2
import numpy as np
import skimage.morphology
import skimage
from matplotlib import pyplot as plt
	
from .getTerminationBifurcation import getTerminationBifurcation
from .removeSpuriousMinutiae import removeSpuriousMinutiae

def extractMinutiae(img, mask):
	
# %% Adaptive thresholding
	print("[thresholding ...]")
	# adaptive thresholding, one can experiment with parameters
	th_img = cv2.adaptiveThreshold(img, 1, 
	                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
	                               cv2.THRESH_BINARY_INV, 11, 2)
	# limit to region inside mask
	th_img = cv2.bitwise_and(th_img, th_img, np.uint8(mask*255))
#	plt.imshow(th_img)
#	plt.show()
	
# %% Thinning
	print("[thinning ...]")
	skel = skimage.morphology.skeletonize(th_img)
	skel = np.uint8(skel)*255;
#	plt.imshow(skel>0)
#	plt.show()
	
# %% Minutiae and bifurcation extraction
	print("[extracting ...]")
	(minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask);
	
# %% Cleaning
	print("[labeling ...]")
	# connected component labeling
	minutiaeTerm = skimage.measure.label(minutiaeTerm, 8);
	# region properties (centroid will be used)
	RP = skimage.measure.regionprops(minutiaeTerm)
	# neighbouring minutiae ending merging
	print("[removing ...]")
	minutiaeTerm = removeSpuriousMinutiae(RP, 10, minutiaeTerm, mask);
	
# %% outputting minutiae as centroids

	BifLabel = skimage.measure.label(minutiaeBif, 8);
	TermLabel = skimage.measure.label(minutiaeTerm, 8);
	
	plot_minutiae = False
	if plot_minutiae:
	
		minutiaeBif = minutiaeBif * 0;
		minutiaeTerm = minutiaeTerm * 0;
		
		(rows, cols) = skel.shape
		DispImg = np.zeros((rows,cols,3), np.uint8)
		DispImg[:,:,0] = skel; DispImg[:,:,1] = skel; DispImg[:,:,2] = skel;
		
		RP = skimage.measure.regionprops(BifLabel)
		for i in RP:
			(row, col) = np.int16(np.round(i['Centroid']))
			minutiaeBif[row, col] = 1;
			(rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
			skimage.draw.set_color(DispImg, (rr,cc), (255,0,0));
	
		RP = skimage.measure.regionprops(TermLabel)
		for i in RP:
			(row, col) = np.int16(np.round(i['Centroid']))
			minutiaeTerm[row, col] = 1;
			(rr, cc) = skimage.draw.circle_perimeter(row, col, 3);
			skimage.draw.set_color(DispImg, (rr,cc), (0, 0, 255));
		cv2.imshow('a',DispImg);
		cv2.waitKey(0)
# %% List the coordinates of bifurcations and endings
			
	RP = skimage.measure.regionprops(BifLabel)
#	minutiaeBifLst = (np.int16(np.round(i['Centroid'])) for i in RP)
	minutiaeBifArr = np.array([ np.int16(np.round(i['Centroid'])) for i in RP ])
	
	RP = skimage.measure.regionprops(TermLabel)
#	minutiaeTermLst = (np.int16(np.round(i['Centroid'])) for i in RP)
	minutiaeTermArr = np.array([ np.int16(np.round(i['Centroid'])) for i in RP ])


	
	return(minutiaeTermArr, minutiaeBifArr, skel)