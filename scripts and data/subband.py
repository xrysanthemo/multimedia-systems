
import numpy as np

def get_impulse_response():
    #read numpy file
    h = np.load('h.npy', allow_pickle=True).item()
    h_coefficients = h["h"]
    return h_coefficients
def make_mp3_analysisfb(h,M):
    H = 0
    return H

def make_mp3_synthesisfb(h,M):
    G = 0
    return G


