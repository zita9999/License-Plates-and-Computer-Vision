
# =============================================================================
# Feature Matching
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL


#function that displays each image
def display(img, cmap = 'gray'):
    fig = plt.figure(figsize = (18,15))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap = 'gray')

#reading in the license plate
license_plate = cv2.imread("/home/asoria/Documents/zita9999/feat_match/license.jpg",0)
display(license_plate)

#reads in the car image
car = cv2.imread("/home/asoria/Documents/zita9999/feat_match/car.png",0)
display(car)


def FAST_n_BRIEF(img1, img2, n=70, threshold=85):
    
    test1 = img1.copy()
    test2 = img2.copy()
    
    #detects the features that are only as strong as the threshold for them
    fast = cv2.FastFeatureDetector_create(threshold = threshold)
    
    kp1=fast.detect(test1,None)

    kp2=fast.detect(test2,None)



    #draws circles on image 1 where features match on image 2
    mark1=cv2.drawKeypoints(test1,kp1,None,color=(0,0,255),flags=0)

    orient1=cv2.drawKeypoints(test1,kp1,None,color=(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    #draws circles on image 2 where features match on image 1
    mark2=cv2.drawKeypoints(test2,kp2,None,color=(255,0,0),flags=0)

    orient2=cv2.drawKeypoints(test2,kp2,None,color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    brisk = cv2.BRISK_create(thresh=75)

    kp1, des1 = brisk.compute(test1, kp1)

    kp2, des2 = brisk.compute(test2, kp2)


    #finds distances between the two matches. Smaller the distance better the match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.match(des1,des2)
 
    matches = sorted(matches, key = lambda x:x.distance)
    
    #draws the lines between the two images
    result = cv2.drawMatches(mark1,kp1,mark2,kp2,matches[:min(n, len(matches))],None,matchColor=[0,255,0], flags=2)

    display(result)
    
    cv2.imwrite("/home/asoria/Documents/zita9999/feat_match/feature_match.jpg",result)
    

FAST_n_BRIEF(license_plate,car)



