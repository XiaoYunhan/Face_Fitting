import numpy as np
import scipy as sp

def quadrilateral_interpolation(pt1, pt2, pt3, pt4, pt):
    # ptx format: (coord_x, coord_y, R, G, B)
    coordination = (pt1[0], pt2[0], pt3[0], pt4[0], pt[0],
            pt1[1], pt2[1], pt3[1], pt4[1], pt[1])
    guess = np.array([0,0])
    [eta, mu] = sp.fsolve(func=find_local, x0=guess, args=coos)
    R = (pt1[2], pt2[2], pt3[2], pt4[2])
    G = (pt1[3], pt2[3], pt3[3], pt4[3])
    B = (pt1[4], pt2[4], pt3[4], pt4[4])
    density_R = density(eta, mu, R)
    density_G = density(eta, mu, G)
    density_B = density(eta, mu, B)

    return (density_R, density_G, density_B)

def find_local(guess):



def density(eta, mu, dsty):
