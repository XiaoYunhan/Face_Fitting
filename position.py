import numpy as np
from scipy.linalg import solve

def position(A, B, C, da, db, dc):
    A11 = 2*A[0] - 2*B[0]
    A12 = 2*A[1] - 2*B[1]
    A21 = 2*A[0] - 2*C[0]
    A22 = 2*A[1] - 2*C[1]
    B01 = A[0]**2+A[1]**2-B[0]**2-B[1]**2+db**2-da**2
    B02 = A[0]**2+A[1]**2-C[0]**2-C[0]**2+dc**2-da**2

    AM = np.array([[A11,A12],[A21,A22]])
    BM = np.array([[B01],[B02]])
    print("AM: ", AM)
    print("BM: ", BM)
    CM = solve(AM,BM)
    print(CM[0][0],CM[1][0])

position((1,2),(2,2),(1,1),1,1.414,2)

