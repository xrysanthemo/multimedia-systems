import numpy as np
from scipy import fft

def frameDCT(Y): #Y einai ena frame N * M
    c = fft.dct(np.ravel(Y, order ='C'))
    return c

def iframeDCT(c): #Ο αντίστροφος μετασχηματισμός DCT
    inverse = fft.idct(c)
    num_of_samples = c.shape[0] // 32
    Yh = inverse.reshape((num_of_samples, 32))
    return Yh