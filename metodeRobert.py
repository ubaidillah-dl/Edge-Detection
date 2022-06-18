import cv2 as cv
import numpy as np

def rescale_frame(frame,percent=75):
    width=int(frame.shape[1]* percent/100)
    height=int(frame.shape[0]* percent/100)
    dim=(width, height)
    return cv.resize(frame,dim,interpolation=cv.INTER_AREA)

camera=cv.VideoCapture(0)

while True:
    _,frame=camera.read()
    image=rescale_frame(frame,percent=70)
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    
    robx=np.array([[ 1, 0],
                   [ 0,-1]])
    roby=np.array([[ 0, 1],
                   [-1, 0]])
    
    img_robx=cv.filter2D(gray,-1,robx)
    img_roby=cv.filter2D(gray,-1,roby)
    robert=cv.add(img_robx,img_roby)
    robert=cv.cvtColor(robert,cv.COLOR_GRAY2BGR)
    
    cv.rectangle(image,(0,0),(150,30),(255,255,255),-1)
    cv.rectangle(robert,(0,0),(150,30),(255,255,255),-1)
    image=cv.putText(image,"Citra Asli",(10,20),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),0)
    robert=cv.putText(robert,"Metode Robert",(10,20),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),0)
    
    stacked=np.vstack((image,robert))
    cv.imshow('Deteksi Tepi',stacked)
    
    if cv.waitKey(1)==ord('p'):
        break

camera.release()
cv.destroyAllWindows()