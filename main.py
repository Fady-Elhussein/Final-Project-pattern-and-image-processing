import cv2

path="person3.jpg"
image=cv2.imread(path)

ratio=500/image.shape[1]
dim=(500,int(image.shape[0]*ratio))
image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)

grey= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

facefile_detect= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faces = facefile_detect.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=5)

eyesfile_detect= cv2.CascadeClassifier("haarcascade_eye.xml")
nosefile_detect= cv2.CascadeClassifier("haarcascade_mcs_nose.xml")


for(x,y,w,h) in faces:
    
    cv2.rectangle(image, (x,y), (x+w,y+h),(255,0,0),3)
    rectangle_face=image[y:y+h,x:x+w] 
    grey_rectangle_face=grey[y:y+h,x:x+w]
    eyes=eyesfile_detect.detectMultiScale(grey_rectangle_face,scaleFactor=1.2,minNeighbors=3)
    nose=nosefile_detect.detectMultiScale(grey_rectangle_face,scaleFactor=3,minNeighbors=3)
    
    for(ex,ey,ew,eh) in eyes:
         for(nx,ny,nw,nh) in nose:
             if(ny > ey):
                cv2.rectangle(rectangle_face,(nx,ny),(nx+nw,ny+nh),(0,255,0),1)
                cv2.rectangle(rectangle_face,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)


cv2.imshow('Detections ', image)
cv2.waitKey(0)
