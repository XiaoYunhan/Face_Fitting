import numpy as np
import scipy as sp

class V2P:
    def __init__(self,pos,q,b1,b2,b3):
        self.pos = np.array([0,0,0,0])
        self.q = q
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

def Vs(ivtx, inst):
    pos = inst.p[ivtx]
    q = pos - inst.p[0]
    b1 = inst.p[1]-inst.p[0]
    b2 = inst.p[2]-inst.p[0]
    b3 = inst.p[0]-inst.p[1]-inst.p[2]+inst.p[3]
    o = new V2P(pos, q, b1, b2, b3)
    return o

def Wedge2D(v, w):
    return v[0]*w[1] - v[1]*w[0]

def Ps(i):
    A = Wedge2D(i.b2, i.b3)
    B = Wedge2D(i.b3, i.q) - Wedge2D(i.b1, i.b2)
    C = Wedge2D(i.b1, i.q)

    uv = np.array([0,0])
    
    if(abs(A) < 0.001):
        uv[1] = -C/B
    else:
        discrim = B*B - 4*A*C
        uv[1] = 0.5*(-B+sqrt(discrim))/A

    denom = i.b1 + uv[1]*i.b3
    if(abs(denom[0]) > abs(denom[1])):
        uv[0] = (i.q[0]-i.b2[0]*uv[1])/denom[0]
    else:
        uv[0] = (i.q[1]-i.b2[1]*uv[1])/denom[1]

    return 
