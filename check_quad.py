import numpy as np

def check_quad(pt0, pt1, pt2, pt3, pt4):
    """This function check whether pt0 is in the quad composed of pt1...4
    pt1...4 are in clockwise
    return: True (in the quad)
            False (not in the quad)"""
    C1 = (pt2[0]-pt1[0])*(pt0[1]-pt1[1])-(pt2[1]-pt1[1])*(pt0[0]-pt1[0])
    C2 = (pt3[0]-pt2[0])*(pt0[1]-pt2[1])-(pt3[1]-pt2[1])*(pt0[0]-pt2[0])
    C3 = (pt4[0]-pt3[0])*(pt0[1]-pt3[1])-(pt4[1]-pt3[1])*(pt0[0]-pt3[0])
    C4 = (pt1[0]-pt4[0])*(pt0[1]-pt4[1])-(pt1[1]-pt4[1])*(pt0[0]-pt4[0])

    print(C1)
    print(C2)
    print(C3)
    print(C4)


check_quad((4,8),(2,6),(6,6),(6,2),(2,2))

#print(result)
