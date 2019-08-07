import cv2
import numpy as np

img = cv2.imread("bill-clinton.jpg")
img2 = cv2.imread("hillary-clinton.jpg")

print(type(img))
print(img.shape)
print(type(img2))
print(img2.shape)
print(img.shape>img2.shape)
print(img)
