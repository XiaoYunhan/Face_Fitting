import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
from imutils import face_utils
import itertools
import dlib
import cv2

class Rbf(object):
    """Radial Basis Function Interpolation
    Kernel function: Thin plate function"""
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
# shape = (68,2)
print("--finished")

print("RBF transformation ...")
height = bill.shape[0]
width = bill.shape[1]
generate_x = np.array([])
generate_y = np.array([])
generate = np.zeros((height, width,2))

#divide image into quad-blocks
sample_vertical = 20
sample_horizontal = 20
sample_coord_x = np.linspace(start=0, stop=width, num=sample_vertical+1, dtype=int)
sample_coord_y = np.linspace(start=0, stop=height, num=sample_horizontal+1, dtype=int)
sample_coord = list(itertools.product(sample_coord_x, sample_coord_y))
sample_coord_block = np.array([[[]]]).reshape(0,4,2)
for i in range(sample_vertical):
    for j in range(sample_horizontal):
        insert = [sample_coord[(sample_horizontal+1)*i+j],
                sample_coord[(sample_horizontal+1)*i+j+1],
                sample_coord[(sample_horizontal+1)*(i+1)+j+1],
                sample_coord[(sample_horizontal+1)*(i+1)+j]]
        insert = np.array(insert)
        sample_coord_block = np.vstack((sample_coord_block,insert[None]))
#print(sample_coord_block)

# RBF deformation on sampling points
sample_coord_block_modified = np.array(sample_coord_block)
hillary_shape_x = hillary_shape[:,0]
hillary_shape_y = hillary_shape[:,1]
bill_shape_x = bill_shape[:,0]
bill_shape_y = bill_shape[:,1]
count_x = 0
count_y = 0
#print(sample_coord)
for sample_index in range((sample_vertical+1)*(sample_horizontal+1)):
    #if sample_index<22 or sample_index%21==20 or sample_index%21==0 or sample_index>420:
        #continue
    x = sample_coord[sample_index][1]
    y = sample_coord[sample_index][0]
    disp_x = np.array([])
    disp_y = np.array([])
    for control_index in range(68):
        di_x = x - hillary_shape[control_index][1]
        di_y = y - hillary_shape[control_index][0]
        disp_x = np.append(disp_x, di_x)
        disp_y = np.append(disp_y, di_y)
    fitting_x = Rbf(hillary_shape_y,hillary_shape_x, disp_x)
    fitting_y = Rbf(hillary_shape_y,hillary_shape_x, disp_y)
    generate_x = np.add(fitting_x(bill_shape_y,bill_shape_x),bill_shape_x)
    generate_y = np.add(fitting_y(bill_shape_y,bill_shape_y),bill_shape_y)
    result_x = np.mean(generate_x)
    result_y = np.mean(generate_y)
    if result_x < 0:
        count_x = count_x+1
    if result_y < 0:
        count_y = count_y+1
    # change coordination in the top left block
    if sample_index>20 and sample_index%21!=0:
        block_index = 20*(sample_index//21-1)+(sample_index%21)-1
        sample_coord_block_modified[block_index][2][0] = result_x
        sample_coord_block_modified[block_index][2][1] = result_y
    # change coordination in the top right block
    if sample_index>20 and sample_index%21!=20:
        block_index = 20*(sample_index//21-1)+(sample_index%21)
        sample_coord_block_modified[block_index][3][0] = result_x
        sample_coord_block_modified[block_index][3][1] = result_y
    # change coordination in the bottom left block
    if sample_index<420 and sample_index%21!=0:
        block_index = 20*(sample_index//21)+(sample_index%21)-1
        sample_coord_block_modified[block_index][1][0] = result_x
        sample_coord_block_modified[block_index][1][1] = result_y
    # change coordination in the bottom right block
    if sample_index<420 and sample_index%21!=20:
        block_index = 20*(sample_index//21)+(sample_index%21)
        sample_coord_block_modified[block_index][0][0] = result_x
        sample_coord_block_modified[block_index][0][0] = result_y

print("--finished")

print(sample_coord_block)
print("////////////////////////////////////////")
print(sample_coord_block_modified)

print(count_x)
print(count_y)









