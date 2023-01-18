import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from scipy.io import wavfile
from dct import iframeDCT, frameDCT
from subband import codec0, get_impulse_response, SNRsystem, coder0, decoder0
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

x_hat, Y_tot = codec0('myfile.wav', h, M, N)

Y_tot2 = coder0('myfile.wav', h, M, N)
x_hat2 = decoder0(Y_tot2, h, M, N)

print("x_hat diff: ", np.mean(np.mean(x_hat - x_hat2)))
print("Y_tot diff: ", np.mean(np.mean(Y_tot - Y_tot2)))

# # #Experiments - SNR
# # SNR = SNRsystem(data, x_hat)
# # print("SNR: ", SNR)
#
# #Πειράματα για DCT
# c = frameDCT(Y_tot[36*50:36*51, :32])
# Y_tot_hat = iframeDCT(c)
#
# #Πειράματα Μέρους 3 - Psychoacoustics
# P = DCTpower(c)
# MN = M * N
# D = Dksparse(MN)
# ST = STinit(c, D)