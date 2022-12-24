
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis

def get_impulse_response():
    #read numpy file
    h = np.load('h.npy', allow_pickle=True).item()
    h_coefficients = h["h"]
    return h_coefficients

#Define the number of subbands
M = 32
h = get_impulse_response().reshape(512,)
H = make_mp3_analysisfb(h, M)
G = make_mp3_synthesisfb(h, M)






