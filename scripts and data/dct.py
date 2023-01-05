import numpy as np
from scipy import fft

def frameDCT(Y): #Y einai ena frame N * M
    #(N, M) = Y.shape  #32 bands, 36 samples
    # #Αρχικοποίηση του διανύσματος c
    # c = np.zeros((N*M, 1))
    #c: Διάσταση NM X1
    #u: συντελεστης μπαντας, i: αριθμος μπαντας k  =iN +u
    #TODO: Ena frequency matrix
    #TODO: Κάπως να είναι flattened
    c = fft.dct(np.ravel(Y, order ='C'))
    return c

def iframeDCT(c):
    inverse = fft.idct(c)
    num_of_samples = c.shape[0] // 32
    Yh = inverse.reshape((num_of_samples, 32))
    return Yh