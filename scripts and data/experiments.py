import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from scipy.io import wavfile
from dct import iframeDCT, frameDCT
from subband import codec0, get_impulse_response, SNRsystem, coder0, decoder0
from psychoacoustics import DCTpower, Dksparse, STinit, MaskPower, get_hearing_threshold, STreduction, Hz2Barks, SpreadFunc, Masking_Thresholds, Global_Masking_Thresholds
from plot import plot_H_Hz, plot_H_barks, plot_err
import matplotlib.pyplot as plt

# Define Parameters
M = 32 #num of filters
L = 512 #len of filters
N = 36 #num of samples
MN = M * N

h = get_impulse_response().reshape(512,)
H = make_mp3_analysisfb(h, M)

sr, data = wavfile.read('myfile.wav')

# # Plot H Hz - barks
# plot_H_Hz(H, sr)
# plot_H_barks(H, sr)

# Codec
x_hat, Y_tot = codec0('myfile.wav', h, M, N)

# # Coder - Decoder
# Y_tot = coder0('myfile.wav', h, M, N)
# x_hat = decoder0(Y_tot, h, M, N)

# print("x_hat diff: ", np.mean(np.mean(x_hat - x_hat2)))
# print("Y_tot diff: ", np.mean(np.mean(Y_tot - Y_tot2)))
# plot_err(data, x_hat)

#Experiments - SNR
SNR = SNRsystem(data, x_hat)
print("SNR: ", SNR)

#Πειράματα για DCT
c = frameDCT(Y_tot)
Y_tot_hat = iframeDCT(c)
# print("Y_tot diff: ", np.mean(np.mean(Y_tot[36*10:36*11] - Y_tot_hat[36*10:36*11])))

# Πειράματα Μέρους 3 - Psychoacoustics
P = DCTpower(c)
D = Dksparse(MN)
ST = STinit(c, D)
print(ST)

# Ισχύς Maskers
PT = MaskPower(c, ST)

# Κατώφλι ακουστότητας
Tq = get_hearing_threshold()

# Ελάτωση των maskers
STr, PMr = STreduction(ST, c, Tq)

# Define Masking Thresholds
Ti = Masking_Thresholds(ST, PMr, MN)

# Define the Global Masking Thresholds
Tg = Global_Masking_Thresholds(Ti, Tq)
plt.plot(Tg)
plt.show()
