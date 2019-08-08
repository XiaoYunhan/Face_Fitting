import cv2
import numpy as np

img = cv2.imread("bill-clinton.jpg")
rows, cols, _ = img.shape
print(rows)
print(cols)

#rotate_matrix = cv2.getRotationMatrix2D((cols/2, rows/2),90,1)

points1 = np.float32([[100,100],[600,0],[0,600],[600,600]])
points2 = np.float32([[0,0],[600,0],[0,600],[600,600]])

matrix = cv2.getPerspectiveTransform(points2,points1)
output = cv2.warpPerspective(img, matrix, (cols,rows))
#img1 = cv2.warpAffine(img, matrix, (cols,rows))

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
#cv2.imshow('img1',img1)
cv2.imshow('output',output)

cv2.waitKey()
cv2.destroyAllWindow()
