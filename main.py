import cv2
import numpy as np
import imutils

path="person7.jpg"
image=cv2.imread(path)
image=imutils.resize(image,500)

grey= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
facefile_detect= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faces = facefile_detect.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=5)
eyesfile_detect= cv2.CascadeClassifier("haarcascade_eye.xml")
mask=np.ones(image.shape[:2], dtype='uint8')*255
mask2=np.zeros(image.shape[:2], dtype='uint8')

for(x,y,w,h) in faces:
    
    cv2.rectangle(image, (x,y), (x+w,y+h),(200,100,50),3)

    cv2.rectangle(mask, (x,y), (x+w,y+h), (0,0,0),-1)
    cv2.rectangle(mask2, (x,y), (x+w,y+h), 255,-1)

    eyes_infaceRectangle=image[y:y+h,x:x+w]
  
    
    grey_eyes_infaceRectangle=grey[y:y+h,x:x+w]
    
    eyes=eyesfile_detect.detectMultiScale(grey_eyes_infaceRectangle)
    
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(eyes_infaceRectangle,(ex,ey),(ex+ew,ey+eh),(0,100,100),3)

masked = cv2.bitwise_and(image, image, mask = mask)
masked2 = cv2.bitwise_and(image, image, mask = mask2)
cv2.imshow("Mask Applied to Image", masked)
cv2.imshow("Mask2 Applied to Image", masked2)
cv2.imshow('Detections ', image)
cv2.waitKey(0)
