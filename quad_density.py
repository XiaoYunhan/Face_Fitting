import numpy as np
from scipy.optimize import fsolve

def interpolate_quatrilateral(pt1,pt2,pt3,pt4,pt):
    '''Interpolates a value in a quatrilateral figure defined by 4 points. 
    Each point is a tuple with 3 elements, x-coo,y-coo and value.
    point1 is the lower left corner, point 2 the lower right corner,
    point 3 the upper right corner and point 4 the upper left corner.
    args is a list of coordinates in the following order:
     x1,x2,x3,x4 and x (x-coo of point to be interpolated) and y1,y2...'''

    coos = (pt1[0],pt2[0],pt3[0],pt4[0],pt[0],
            pt1[1],pt2[1],pt3[1],pt4[1],pt[1])
    guess = np.array([0,0])
    [eta, mu] = fsolve(func=find_local_coo_equations, x0=guess, args=coos)

    densities = (pt1[2], pt2[2], pt3[2], pt4[2])
    density = find_density(eta,mu,densities)

    return density

def find_local_coo_equations(guess, *args):
    '''This function creates the transformed coordinate equations of the quatrilateral.'''

    eta = guess[0]
    mu = guess[1]

    eq=[0,0]#Initialize eq
    eq[0] = 1 / 4 * (args[0] + args[1] + args[2] + args[3]) - args[4] + \
            1 / 4 * (-args[0] - args[1] + args[2] + args[3]) * mu + \
            1 / 4 * (-args[0] + args[1] + args[2] - args[3]) * eta + \
            1 / 4 * (args[0] - args[1] + args[2] - args[3]) * mu * eta
    eq[1] = 1 / 4 * (args[5] + args[6] + args[7] + args[8]) - args[9] + \
            1 / 4 * (-args[5] - args[6] + args[7] + args[8]) * mu + \
            1 / 4 * (-args[5] + args[6] + args[7] - args[8]) * eta + \
            1 / 4 * (args[5] - args[6] + args[7] - args[8]) * mu * eta
    return eq

def find_density(eta,mu,densities):
    '''Finds the final density based on the eta and mu local coordinates calculated
    earlier and the densities of the 4 points'''
    N1 = 1/4*(1-eta)*(1-mu)
    N2 = 1/4*(1+eta)*(1-mu)
    N3 = 1/4*(1+eta)*(1+mu)
    N4 = 1/4*(1-eta)*(1+mu)
    density = densities[0]*N1+densities[1]*N2+densities[2]*N3+densities[3]*N4
    return density

pt1= (0,0,1)
pt2= (1,0,1)
pt3= (1,1,2)
pt4= (0,1,2)
pt= (0.5,0.5)
print(interpolate_quatrilateral(pt1,pt2,pt3,pt4,pt))
