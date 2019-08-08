import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
from imutils import face_utils
import dlib
import cv2

class Rbf(object):
        def thin_plate(self, r):
                return xlogy(r**2, r)
        def __init__(self, input_x, input_y, displacement):
                self.x = input_x
                self.y = input_y
                self.d = displacement
                self.flatten = np.asarray([np.asarray(self.x).flatten(),
                    np.asarray(self.y).flatten()])
                self.num_of_input = self.flatten.shape[-1]
                self.last = np.asarray(self.d).flatten()
                self.A = self.thin_plate(squareform(pdist(self.flatten.T,'euclidean')))
                self.B = linalg.solve(self.A, self.last)
        def __call__(self, input_x, input_y):
                sp = input_x.shape
                xa = np.asarray([input_x.flatten(), input_y.flatten()])
                r = cdist(xa.T, self.flatten.T, 'euclidean')
                return np.dot(self.thin_plate(r), self.B).reshape(sp)

print("preprocessing ...")
bill = cv2.imread("bill-clinton.jpg")
hillary = cv2.imread("hillary-clinton.jpg")
#resize images with interpolation
if bill.shape!=hillary.shape:
    if bill.shape>hillary.shape:
        dim = (bill.shape[1],bill.shape[0])
        hillary = cv2.resize(hillary, dim, interpolation=cv2.INTER_AREA)
    else:
        dim = (hillary.shape[1],hillary.shape[0])
        bill = cv2.resize(hillary, dim, interpolation=cv2.INTER_AREA)
print("--finished")

print("face landmarks initialization ...")
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

bill_gray = cv2.cvtColor(bill, cv2.COLOR_BGR2GRAY)
hillary_gray = cv2.cvtColor(bill, cv2.COLOR_BGR2GRAY)
bill_rects = detector(bill_gray, 0)
hillary_rects = detector(hillary_gray, 0)

for(i, rect) in enumerate(bill_rects):
    bill_shape = predictor(bill_gray, rect)
    bill_shape = face_utils.shape_to_np(bill_shape)

for(i, rect) in enumerate(hillary_rects):
    hillary_shape = predictor(hillary_gray,rect)
    hillary_shape = face_utils.shape_to_np(hillary_shape)

#print(bill_shape.shape)
#print(hillary_shape.shape)
print("--finished")

print("RBF deformation ...")

#print(bill.shape)
#print(hillary.shape)
height = bill.shape[0]
width = bill.shape[1]
generate_x = np.array([])
generate_y = np.array([])
result_x = np.array([])
result_y = np.array([])
generate = np.zeros((height,width,2))
for pixel in range(height*width):
    y = pixel/width
    x = pixel%width
    disp_x = np.array([])
    disp_y = np.array([])
    for control_index in range(68):
        di_x = x - hillary_shape[control_index][1]
        di_y = y - hillary_shape[control_index][0]
        disp_x = np.append(disp_x, di_x)
        disp_y = np.append(disp_y, di_y)
    fitting_x = Rbf(hillary_shape[:,1],
            hillary_shape[:,0], disp_x)
    fitting_y = Rbf(hillary_shape[:,1],
            hillary_shape[:,0], disp_y)
    #print(bill_shape[:,0].shape)
    #print(bill_shape[:,1].shape)
    #print(fitting_x(bill_shape[:,1],bill_shape[:,0]).shape)
    generate_x = np.add(fitting_x(bill_shape[:,1],
        bill_shape[:,0]),bill_shape[:,0])
    generate_y = np.add(fitting_y(bill_shape[:,1],
        bill_shape[:,0]),bill_shape[:,1])
    result_x = np.append(result_x, np.mean(generate_x))
    result_y = np.append(result_y, np.mean(generate_y))


print("--finished")








