import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import string
import math

PITCH = -32.7  # points down ugh
YAW = -24.5  # I believe relative to magnetic north
ROLL = -13  # might not need


def canny_edge_detector(image):
    # Convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Reduce noise from the image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def R_z(psi):
    radians = math.radians(psi)
    return [[math.cos(radians), -math.sin(radians), 0], [math.sin(radians), math.cos(radians), 0], [0, 0, 1]]


def R_y(theta):
    radians = math.radians(theta)
    return [[math.cos(radians), 0, math.sin(radians)], [0, 1, 0], [-math.sin(radians), 0, math.cos(radians)]]


def R_x(phi):
    radians = math.radians(phi)
    return [[1, 0, 0], [0, math.cos(radians), -math.sin(radians)], [0, math.sin(radians), math.cos(radians)]]


image = cv2.imread('IMG_0198.JPG')
(h, w) = (image.shape[0], image.shape[1])
R = np.matmul(np.matmul(R_x(-13), R_y(-32.7)), R_z(0))
outImage = cv2.warpPerspective(image, R, (w, h))


random = ''.join([random.choice(string.ascii_letters + string.digits)
                  for n in range(10)])
print(random)
cv2.imwrite("transformed/" + random + ".jpg", outImage)
