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
    gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    soby=np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    sobx=np.array([[ 1, 2, 1],
                   [ 0, 0, 0],
                   [-1,-2,-1]])
    
    robx=np.array([[ 1, 0],
                   [ 0,-1]])
    roby=np.array([[ 0, 1],
                   [-1, 0]])
    
    prex=np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    prey=np.array([[ 1, 1, 1],
                   [ 0, 0, 0],
                   [-1,-1,-1]])
    
    img_sobx=cv.filter2D(gray,-1,sobx)
    img_soby=cv.filter2D(gray,-1,soby)
    sobel=cv.add(img_sobx, img_soby)
    
    img_robx=cv.filter2D(gray,-1,robx)
    img_roby=cv.filter2D(gray,-1,roby)
    robert=cv.add(img_robx,img_roby)
    
    img_prex=cv.filter2D(gray,-1, prex)
    img_prey=cv.filter2D(gray,-1, prey)
    prewitt=cv.add(img_prex,img_prey)
    
    cv.rectangle(image,(0,0),(150,30),(255,255,255),-1)
    cv.rectangle(sobel,(0,0),(150,30),(255,255,255),-1)
    cv.rectangle(robert,(0,0),(150,30),(255,255,255),-1)
    cv.rectangle(prewitt,(0,0),(150,30),(255,255,255),-1)
    image=cv.putText(image,"Citra Asli",(10,20),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),0)
    sobel=cv.putText(sobel,"Metode Sobel",(10,20),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),0)
    robert=cv.putText(robert,"Metode Robert",(10,20),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),0)
    prewitt=cv.putText(prewitt,"Metode Prewitt",(10,20),cv.FONT_HERSHEY_PLAIN,1,(0,0,0),0)
    
    sobel=cv.cvtColor(sobel,cv.COLOR_GRAY2BGR)
    robert=cv.cvtColor(robert,cv.COLOR_GRAY2BGR)
    prewitt=cv.cvtColor(prewitt,cv.COLOR_GRAY2BGR)
    
    row1=np.hstack((image,sobel))
    row2=np.hstack((robert,prewitt))
    
    stacked=np.vstack((row1,row2))
    cv.imshow('Deteksi Tepi',stacked)
    
    if cv.waitKey(1)==ord('p'):
        break

camera.release()
cv.destroyAllWindows()