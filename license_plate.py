# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL

#reading in the input image
plate = cv2.imread("C:/Users/chris/OneDrive/Pictures/car7.jpg")

#function that shows the image
def display(img, cmap = 'gray'):
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap = 'gray')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("C:/Users/chris/OneDrive/Pictures/main_car4.jpg",img)

#need to change color of picture from BGR to RGB
plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
display(plate)

#Cascade Classifier where our hundres of samples of license plates are
plate_cascade = cv2.CascadeClassifier('C:/Users/chris/OneDrive/Documents/Courses and Projects/Computer Vision/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_russian_plate_number.xml')


def detect_plate(img):
    
    plate_img = plate.copy()
    
    #gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 10)

    #draws the rectangle around it
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (255,0,0), 5)

    return plate_img

result = detect_plate(plate)
display(result)


#detects the plate and zooms in on it
def detect_zoom_plate(img, kernel):
    
    plate_img = img.copy()
    
    #gets the points of where the classifier detects a plate
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 10) #maxSize = (100,100))
    
    for (x,y,w,h) in plate_rects:
        x_offset = x
        y_offset = y
        
        x_end = x+w
        y_end = y+h
        
        #getting the points that show the license plate
        zoom_img = plate_img[y_offset:y_end, x_offset:x_end]
        #increasing the size of the image
        zoom_img = cv2.resize(zoom_img, (0,0),fx = 2, fy = 2)
        zoom_img = zoom_img[7:-7, 7:-7]
        #sharpening the image to make it look clearer
        zoom_img = cv2.filter2D(zoom_img, -1, kernel)
        
        zy = (40 - (y_end - y_offset))//2
        zx = (144 - (x_end-x_offset))//2
        
        ydim = (y_end+zy-50) - (y_offset-zy-50)
        xdim = (x_end+zx) - (x_offset-zx)
       
       
        zoom_img = cv2.resize(zoom_img,(xdim,ydim))
        
        #putting the zoomed in image above where the license plate is located
        try:
            plate_img[y_offset-zy-55:y_end+zy-55, x_offset-zx:x_end+zx] = zoom_img
        except:
            pass
         
        #drawing a rectangle
        for (x,y,w,h) in plate_rects:
            cv2.rectangle(plate_img, (x,y), (x+w, y+h), (255,0,0), 2)
            
        
    return plate_img

#same function as above just blurs the license plate instead
def detect_blur(img):
    
    plate_img = img.copy()
    
    
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 3)
    
    for (x,y,w,h) in plate_rects:
        x_offset = x
        y_offset = y
        
        x_end = x+w
        y_end = y+h
        
        zoom_img = plate_img[y_offset:y_end, x_offset:x_end]
        #blur function
        zoom_img = cv2.medianBlur(zoom_img,15)
        plate_img[y_offset:y_end, x_offset:x_end] = zoom_img
        
        for (x,y,w,h) in plate_rects:
            cv2.rectangle(plate_img, (x,y), (x+w, y+h), (255,0,0), 5)
        
    return plate_img
    
#matrix needed to sharpen the image
kernel = np.array([[-1,-1,-1],
                   [-1,9,-1],
                   [-1,-1,-1]])
    
result = detect_zoom_plate(plate, kernel)
display(result)

result = detect_blur(plate)
display(result)




#### video
cap = cv2.VideoCapture('C:/Users/chris/OneDrive/Documents/Apowersoft/Video Editor Pro/Output/MyVideo_5.mp4')

#gets the height and width of each frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#saves the video to a file
writer = cv2.VideoWriter('C:/Users/chris/OneDrive/Documents/Courses and Projects/Computer Vision/vid_zoom2.mp4', cv2.VideoWriter_fourcc(*'DIVX'),20,(width, height))

#to make video actual speed
if cap.isOpened() == False:
    print('error file not found')

#while the video is running the loop will keep running
while cap.isOpened():
    #returns each frame
    ret, frame = cap.read()
    
    # if there are still frames keeping showing the video
    if ret == True:
        #apply our detect and zoom function to each frame
        frame = detect_zoom_plate(frame, kernel)
        #show the frame
        cv2.imshow('frame', frame)
        writer.write(frame)
        
        #will stop the video if it fnished or you press q
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    else:
        break

#stop the video, and gets rid of the window that it opens up        
cap.release()
writer.release()
cv2.destroyALLWindows()









