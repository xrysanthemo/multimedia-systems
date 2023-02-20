import numpy as np
from scipy import fft

def frameDCT(Y: np.ndarray) ->np.ndarray: #Y είναι ένα frame N * M
    """
    Υλοποίηση του DCT μετασχηματισμού για ένα frame
    """
    c = 2 * fft.dct(np.ravel(Y, order ='C'), norm= 'ortho')
    return c

def iframeDCT(c:np.ndarray) -> np.ndarray: #Ο αντίστροφος μετασχηματισμός DCT
    """
    Αντίστροφος DCT μετασχηματισμός για ένα frame
    """
    inverse = (1/2) * fft.idct(c, norm='ortho')
    num_of_samples = c.shape[0] // 32
    Yh = inverse.reshape((num_of_samples, 32))
    return Yh