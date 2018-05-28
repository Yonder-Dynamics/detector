import numpy as np
import cv2
import glob

for imagePath in glob.glob("test_images/*.JPEG"):
    image = cv2.imread(imagePath, 0)
    image = cv2.medianBlur(image, 5)
    cv2.imshow("Test", image)
    cv2.waitKey(0)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
            param1=3,param2=120,minRadius=0,maxRadius=0)
    print(circles)
    circles = np.uint16(np.around(circles))
    cimg = image
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
