import math

def cal_ratio(pt0,pt1,pt2,pt3,pt4):
    """In quad ABCD, we return the ratio
    u = AE/AB = DF/DC
    v = EO/EF
    vertex A,B,C,D => pt1...4, point O pt0
    quadratic equation form: ax^2+bx+c=0
    """
    a = (pt2[0]-pt1[0])*(pt3[0]-pt4[0])+(pt2[1]-pt1[1])*(pt3[1]-pt4[1])
    b = (pt1[0]-pt2[0])*(pt0[0]-pt4[0])+(pt0[0]-pt1[0])*(pt4[0]-pt3[0])+\
            (pt0[1]-pt1[1])*(pt4[1]-pt3[1])+(pt1[1]-pt2[1])*(pt0[1]-pt4[1])
    c = (pt0[0]-pt1[0])*(pt0[0]-pt4[0])+(pt0[1]-pt1[1])*(pt0[1]-pt4[1])
    
    delta = b**2-4*a*c
    if d<0:
        print "Point 0 is not in this quad"
        return false
    elif d==0:
        u = (-b)/(2*a)
    else:
        u1 = (-b+math.sqrt(delta))/(2*a)
        u2 = (-b-math.sqrt(delta))/(2*a)
        if u1>0:
            u = u1
        else:
            u = u2

    EFx = abs(pt4[0]-pt1[0]+u*(pt3[0]+pt1[0]-pt4[0]-pt2[0]))
    EOx = abs(pt0[0]-pt1[0]+u*(pt1[0]-pt2[0]))
    v = EOx/EFx

    return (u,v)

