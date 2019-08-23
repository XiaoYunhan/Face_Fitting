import numpy as np
import cv2

image = cv2.imread("bill-clinton.jpg")

height = image.shape[0]
width = image.shape[1]

copy = np.array(image)
output = 255*np.ones(image.shape, dtype=image.dtype)

#print(height)
#print(width)

tri1 = np.float32([[[360,200],[60,250],[450,400]]])
tri2 = np.float32([[[400,200],[160,270],[400,400]]])

bounding1 = cv2.boundingRect(tri1)
bounding2 = cv2.boundingRect(tri2)

tri1Cropped = []
tri2Cropped = []

for i in range(3):
    tri1Cropped.append(((tri1[0][i][0]-bounding1[0]),(tri1[0][i][1]-bounding1[1])))
    tri2Cropped.append(((tri2[0][i][0]-bounding2[0]),(tri2[0][i][1]-bounding2[1])))

img1Cropped = image[bounding1[1]:bounding1[1]+bounding1[3], bounding1[0]:bounding1[0]+bounding1[2]]
warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

img2Cropped = cv2.warpAffine(img1Cropped,warpMat,(bounding2[2],bounding2[3]),
        None,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

mask = np.zeros((bounding2[3], bounding2[2], 3), dtype=np.float32)
cv2.fillConvexPoly(mask,np.int32(tri2Cropped),(1.0,1.0,1.0),16,0);
img2Cropped = img2Cropped*mask

output[bounding2[1]:bounding2[1]+bounding2[3], bounding2[0]:bounding2[0]+bounding2[2]] = \
        output[bounding2[1]:bounding2[1]+bounding2[3], 
        bounding2[0]:bounding2[0]+bounding2[2]]*((1.0,1.0,1.0)-mask)

output[bounding2[1]:bounding2[1]+bounding2[3], bounding2[0]:bounding2[0]+bounding2[2]] = \
        output[bounding2[1]:bounding2[1]+bounding2[3],bounding2[0]:bounding2[0]+ \
        bounding2[2]]+img2Cropped

cv2.imwrite("affine.jpg", output)
