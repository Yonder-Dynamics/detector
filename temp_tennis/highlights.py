import cv2
import numpy as np
import glob

# B G R

upper_bright = np.array([255, 255, 255])
lower_bright = np.array([0, 0, 0])
# Bright blue
upper_bright[0] = 100
lower_bright[0] = 0
# Bright green
upper_bright[1] = 255
lower_bright[1] = 130
# Bright red
upper_bright[2] = 150
lower_bright[2] = 100

for imagePath in glob.glob("test_images/*.JPEG"):
    image = cv2.imread(imagePath)
    image = cv2.blur(image, (5,5))
    std = np.std(image)
    mean = np.mean(image)
    #image -= int(mean) + 128

    mask = cv2.inRange(image, lower_bright, upper_bright)# / 255
    cv2.imshow("Mask", mask)
    char = chr(cv2.waitKey(0) & 0xFF)
    ys = np.expand_dims(np.arange(0, image.shape[0]), 1).dot(np.ones((1, image.shape[1])))
    xs = np.ones((image.shape[0], 1)).dot(np.expand_dims(np.arange(0, image.shape[1]), 0))
    x = (np.sum(np.multiply(ys, mask)) / max(np.count_nonzero(mask), 1))
    y = (np.sum(np.multiply(xs, mask)) / max(np.count_nonzero(mask), 1))
    cv2.circle(image, (int(y), int(x)), 10, (255, 255, 255))
    cv2.imshow("Mask", image)
    if chr(cv2.waitKey(0) & 0xFF) == "q":
        break

