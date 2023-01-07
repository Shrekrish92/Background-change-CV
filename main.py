import cv2 as cv
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np

segment=SelfiSegmentation()
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
listImg=os.listdir("Images")
print(listImg)
imglist=[]
indexImg=1
for imgPath in listImg:
    img=cv.imread(f'Images/{imgPath}')
    imglist.append(img)
    print(np.array(img).shape)
print(len(imglist))
imgBG=cv.imread("Images/1.jpg")

fpsread=cvzone.FPS()

while True:
    success, img =  cap.read()
    imgout=segment.removeBG(img, imglist[indexImg], threshold=0.8)

    imgstack=cvzone.stackImages([img,imgout],2,1)
    _, imgstack=fpsread.update(imgstack)
    cv.imshow('camera',imgstack)
    key=cv.waitKey(1)

    if key== ord('a'):
        indexImg -= 1
    elif key== ord('d'):
        indexImg += 1
    elif key== ord('q'):
        break   
