# import the necessary packages
import numpy as np
import imutils
import glob
import cv2

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread("32.jpg")
template_g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_c = cv2.Canny(template, 30, 170)
vis = True
(tH, tW) = template.shape[:2]
if vis:
    cv2.imshow("Template", template_c)
    cv2.waitKey(0)

# loop over the images to find the template in
for imagePath in glob.glob("test_images/*.JPEG"):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    cv2.imshow("Test", image)
    cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    edged = cv2.Canny(gray, 50, 200)
    # loop over the scales of the image
    for scale in np.linspace(1, 0.02, 40)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        template_resized = imutils.resize(template, width = int(template.shape[1] * scale))
        r = template.shape[1] / float(template_resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if template_resized.shape[0] > gray.shape[0] or template_resized.shape[1] > gray.shape[1]:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        template_resized_c = cv2.Canny(template_resized, 50, 200)
        result = cv2.matchTemplate(edged, template_resized_c, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        maxVal = maxVal * r
 
        # check to see if the iteration should be visualized
        if vis:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + template_resized.shape[0], maxLoc[1] + template_resized.shape[1]), (0, 0, 255), 2)
            print(maxVal)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)
 
        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r, 0, template_resized.shape[0], template_resized.shape[1])

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template_c, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 
        # check to see if the iteration should be visualized
        if vis:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)
 
        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r, 1, 0, 0)
 
    # unpack the bookkeeping varaible and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    print found
    (_, maxLoc, r, mode, h, w) = found
    if mode == 0:
        (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
        (endX, endY) = (int(maxLoc[0] + h), int(maxLoc[1] + w))
    elif mode == 1:
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
 
    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
