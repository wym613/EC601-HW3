import numpy as np
import cv2
img = imread('/Users/wangyimeng/Desktop/好好学习天天向上/EC601/HW3/OpenCV_homework/Test_images/Lenna.png')
cv2.namedWindow('Original image')
cv2.imshow('Original image',img)
b,g,r = cv2.split(img)
cv2.imshow("Blue",b)
cv2.imshow("Green",g)
cv2.imshow("Red".r)
cv2.imwrite("Blue.png",b)
cv2.imwrite("Green.png",g)
cv2.imwrite("Red.png",r)

#YCrCb
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y,cb,cr = cv2.split(img_YCrCb)
cv2.imshow("Y",y)
cv2.imshow("Cb",cb)
cv2.imshow("Cr",cr)
cv2.imwrite("Y,png",y)
cv2.imwrite("Cb.png",cb)
cv2.imwrite("Cr.png",cr)

#hsv
img_hsv = cv2.cvtColor(img,cv2.COLOR_BG2HSV)
hue,saturation,value = split(img_hsv)
cv2.imshow("Hue".hue)
cv2.imshow("Saturation".saturation)
cv2.imshow("Value".value)
cv2.imwrite("Hue.png".hue)
cv2.imwrite("Saturation.png".saturation)
cv2.imwrite("Value.png".value)

cv2.waitKey(0)
cv2.destroyAllWindows()
