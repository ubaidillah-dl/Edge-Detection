import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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
    
    sobx=np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    soby=np.array([[ 1, 2, 1],
                   [ 0, 0, 0],
                   [-1,-2,-1]])
    
    img_sobx=cv.filter2D(gray,-1,sobx)
    img_soby=cv.filter2D(gray,-1,soby)
    sobel=cv.add(img_sobx,img_soby)
    sobel=cv.cvtColor(sobel,cv.COLOR_GRAY2BGR)
    
    cv.rectangle(image,(0,0),(150,30),(255,255,255),-1)
    cv.rectangle(sobel,(0,0),(150,30),(255,255,255),-1)
    image=cv.putText(image,"Citra Asli",(10,20),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),0)
    sobel=cv.putText(sobel,"Metode Sobel",(10,20),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),0)
    
    stacked=np.vstack((image,sobel))
    cv.imshow('Deteksi Tepi',stacked)
    
    if cv.waitKey(1)==ord('p'):
        break

camera.release()
cv.destroyAllWindows()