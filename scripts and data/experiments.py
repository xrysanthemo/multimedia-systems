import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from scipy.io import wavfile
from dct import iframeDCT, frameDCT
from subband import codec0, get_impulse_response, SNRsystem
from psychoacoustics import DCTpower, Dksparse, STinit
from plot import plot_H_Hz, plot_H_barks

# Define Parameters
M = 32 #num of filters
L = 512 #len of filters
N = 36 #num of samples

h = get_impulse_response().reshape(512,)
H = make_mp3_analysisfb(h, M)

sr, data = wavfile.read('myfile.wav')

# # Plot H Hz - barks
# plot_H_Hz(H, sr)
# plot_H_barks(H, sr)
#
x_hat, Y_tot = codec0('myfile.wav', h, M, N)
#
# #Experiments - SNR
# SNR = SNRsystem(data, x_hat)
# print("SNR: ", SNR)

#Πειράματα για DCT
c = frameDCT(Y_tot)
Y_tot_hat = iframeDCT(c)

#Πειράματα Μέρους 3 - Psychoacoustics

P = DCTpower(c)
size = M *N
D = Dksparse(size)
print(D)
ST = STinit(c,D)