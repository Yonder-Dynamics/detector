import cv2
import numpy as np
import glob

# B G R
#upper = np.array([100, 234, 207])
#lower = np.array([7, 77, 108])
#upper_bright = np.array([200, 255, 255])
#lower_bright = np.array([150, 200, 0])

upper_bright = np.array([255, 255, 255])
lower_bright = np.array([0, 0, 0])
# Bright blue
upper_bright[0] = 200
lower_bright[0] = 103
# Bright green
upper_bright[1] = 255
lower_bright[1] = 165
# Bright red
upper_bright[2] = 255
lower_bright[2] = 148

upper_shade = np.array([255, 255, 255])
lower_shade = np.array([0, 0, 0])
# Shade blue
upper_shade[0] = 103
lower_shade[0] = 0
# Shade green
upper_shade[1] = 167
lower_shade[1] = 10
# Shade red
upper_shade[2] = 136
lower_shade[2] = 9



for imagePath in glob.glob("test_images/*.JPEG"):
    image = cv2.imread(imagePath)
    image = cv2.blur(image, (5,5))
    std = np.std(image)
    mean = np.mean(image)
    #image -= int(mean) + 128

    mask = cv2.inRange(image, lower_shade, upper_bright)# / 255
    mask1 = cv2.inRange(image, lower_bright, upper_bright)# / 255
    mask2 = cv2.inRange(image, lower_shade, upper_shade)# / 255
    cv2.imshow("Mask", mask)
    char = chr(cv2.waitKey(0) & 0xFF)
    """
    ys = np.expand_dims(np.arange(0, image.shape[0]), 1).dot(np.ones((1, image.shape[1])))
    xs = np.ones((image.shape[0], 1)).dot(np.expand_dims(np.arange(0, image.shape[1]), 0))
    x = (np.sum(np.multiply(ys, mask)) / max(np.count_nonzero(mask), 1))
    y = (np.sum(np.multiply(xs, mask)) / max(np.count_nonzero(mask), 1))
    cv2.circle(image, (int(y), int(x)), 10, (255, 255, 255))
    while char != "q":
        mask = cv2.inRange(image, lower_bright, upper_bright)# / 255
        mask1 = cv2.inRange(image, lower_bright, upper_bright)# / 255
        mask2 = cv2.inRange(image, lower_bright, upper_bright)# / 255
        if char == "a":
            upper_bright[0] -= 1
        if char == "d":
            upper_bright[0] += 1

        cv2.imshow("Mask", mask1)
        char = chr(cv2.waitKey(0) & 0xFF)

    print(upper_bright)
    upper_bright[0] = 255
    """
    masked = cv2.bitwise_and(image,image,mask = cv2.medianBlur(mask, 5))

    cv2.imshow("Mask", masked)
    if chr(cv2.waitKey(0) & 0xFF) == "q":
        break
