import numpy as np


def frameDCT(Y): #Y einai ena frame N * M
    (N, M) = Y.shape
    #Αρχικοποίηση του διανύσματος c
    c = np.zeros((N*M, 1))
    #c: Διάσταση NM X1
    #TODO: Ena frequency matrix
    return c

def iframeDCT(c):
    return Yh