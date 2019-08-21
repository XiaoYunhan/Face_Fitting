import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
from imutils import face_utils
import itertools
import dlib
import cv2
import math

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

def check_quad(pt0, pt1, pt2, pt3, pt4):
    """This function check whether pt0 is in the quad composed of pt1...4
    pt1...4 are in clockwise
    return: True (in the quad)
            False (not in the quad)"""
    C1 = (pt2[0]-pt1[0])*(pt0[1]-pt1[1])-(pt2[1]-pt1[1])*(pt0[0]-pt1[0])
    C2 = (pt3[0]-pt2[0])*(pt0[1]-pt2[1])-(pt3[1]-pt2[1])*(pt0[0]-pt2[0])
    C3 = (pt4[0]-pt3[0])*(pt0[1]-pt3[1])-(pt4[1]-pt3[1])*(pt0[0]-pt3[0])
    C4 = (pt1[0]-pt4[0])*(pt0[1]-pt4[1])-(pt1[1]-pt4[1])*(pt0[0]-pt4[0])

    result = (C1>0 and C2>0 and C3>0 and C4>0) or (C1<0 and C2<0 and C3<0 and C4<0)

    return C1 and C2 and C3 and C4

def bi_interp_ratio(pt0, pt1, pt2, pt3, pt4):
    """In quad ABCD (clockwise), we return the ratio
    u = AE/AB = DF/DC (0<u<1)
    v = EO/EF (0<v<1)
    vertex A,B,C,D => pt1...4, point O pt0
    quadratic equation form: ax^2+bx+c=0
    """
    a = (pt2[0]-pt1[0])*(pt3[1]-pt4[1])-(pt3[0]-pt4[0])*(pt2[1]-pt1[1])
    b = -(pt0[0]-pt1[0])*(pt3[1]-pt4[1])-(pt2[0]-pt1[0])*(pt0[1]-pt4[1])+\
            (pt0[0]-pt4[0])*(pt2[1]-pt1[1])+(pt3[0]-pt4[0])*(pt0[1]-pt1[1])
    c = (pt0[0]-pt1[0])*(pt0[1]-pt4[1])-(pt0[1]-pt1[1])*(pt0[0]-pt4[0])
    
    if a==0:
        u = -c/b
    else:
        delta = b**2-4*a*c
        if delta<0:
            print("Point 0 is not in this quad")
            return False
        elif delta==0:
            u = (-b)/(2*a)
        else:
            u1 = (-b+math.sqrt(delta))/(2*a)
            u2 = (-b-math.sqrt(delta))/(2*a)
            if u1>0:
                u = u1
            else:
                u = u2

    EFy = pt4[1]-pt1[1]+u*(pt3[1]+pt1[1]-pt4[1]-pt2[1])
    EOy = pt0[1]-pt1[1]+u*(pt1[1]-pt2[1])
    v = EOy/EFy

    return (u,v)

def quad_interpolation(pt0, pt1, pt2, pt3, pt4, tg1, tg2, tg3, tg4):
    """In this function, we pass coordinations of quad vertices pt1...4
    and target quad vertices tg1...4
    return: the corresponding coordination of pt0 in the original image"""
    u, v = bi_interp_ratio(pt0, pt1, pt2, pt3, pt4)
    x = int(tg1[0] + u*(tg2[0]-tg1[0]))
    y = int(tg1[1] + v*(tg4[1]-tg1[1]))
    return (x,y)

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
# shape = (68,2)
print("--finished")

print("RBF deformation ...")

height = bill.shape[0]
width = bill.shape[1]
generate_x = np.array([])
generate_y = np.array([])

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
sample_coord_block_modified = sample_coord_block
hillary_shape_x = hillary_shape[:,0]
hillary_shape_y = hillary_shape[:,1]
bill_shape_x = bill_shape[:,0]
bill_shape_y = bill_shape[:,1]

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

print("image warping ...")

output_image = np.zeros(bill.shape)

for pixel in range(width*height):
    x = pixel%width
    y = pixel//width
    current_block = -1
    #find the corresponding block
    for block in range(sample_vertical*sample_horizontal):
        if check_quad((x,y), tuple(sample_coord_block_modified[block][0]), \
                tuple(sample_coord_block_modified[block][1]), \
                tuple(sample_coord_block_modified[block][2]), \
                tuple(sample_coord_block_modified[block][3])):
            current_block = block
            break

    if current_block!=-1:
        #find the corresponding pixel the original image
        c_pixel = quad_interpolation((x,y), \
                tuple(sample_coord_block_modified[current_block][0]), \
                tuple(sample_coord_block_modified[current_block][1]), \
                tuple(sample_coord_block_modified[current_block][2]), \
                tuple(sample_coord_block_modified[current_block][3]), \
                tuple(sample_coord_block[current_block][0]), \
                tuple(sample_coord_block[current_block][1]), \
                tuple(sample_coord_block[current_block][2]), \
                tuple(sample_coord_block[current_block][3]))
        #assign the corresponding RGB color to pixel
        output_image[y][x][0] = bill[c_pixel[1]][c_pixel[0]][0]
        output_image[y][x][1] = bill[c_pixel[1]][c_pixel[0]][1]
        output_image[y][x][2] = bill[c_pixel[1]][c_pixel[0]][2]


print("--finished")

print("output processing ...")

# processing edge of image

# output as jpg image file
cv2.imwrite("output.jpg", out_image)

print("--finished")














