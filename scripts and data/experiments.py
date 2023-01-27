import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from scipy.io import wavfile
from dct import iframeDCT, frameDCT
from subband import codec0, get_impulse_response, SNRsystem, coder0, decoder0
from psychoacoustics import DCTpower, Dksparse, STinit, MaskPower, get_hearing_threshold, STreduction, Hz2Barks, psycho
from quantization import critical_bands, DCT_band_scale, quantizer, dequantizer
from plot import plot_H_Hz, plot_H_barks, plot_err, plot_snr
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
#
# print("x_hat diff: ", np.mean(np.mean(x_hat - x_hat2)))
# print("Y_tot diff: ", np.mean(np.mean(Y_tot - Y_tot2)))
# plot_err(data, x_hat)

#Experiments - SNR
SNR = SNRsystem(data, x_hat)
# signal = np.mean(np.float64(data)**2)
# noise = (np.mean(np.float64(data) - np.mean(np.float64(x_hat)))**2)
# SNR = 10*np.log10(signal/noise)
print("SNR: ", SNR, " dB")

# plot_snr(data, x_hat)

# error = data - x_hat
# plot_err(data, x_hat)


#Πειράματα για DCT
c = frameDCT(Y_tot)
Y_tot_hat = iframeDCT(c)
# print("Y_tot diff: ", np.mean(np.mean(Y_tot[36*10:36*11] - Y_tot_hat[36*10:36*11])))

# Πειράματα Psychoacoustics
# Υπολογισμος συχνοτικών περιοχών
D = Dksparse(MN)
# Υπολογισμός κατωφλίου ακουστότητας
Tg = psycho(c, D)
# plt.plot(Tg)
# plt.show()

# Πειράματα Quantization
# Scale DCT
cs, sc = DCT_band_scale(c[MN*0:MN*1])
# bits num
b = 4
# Quantize
symb_index = quantizer(cs, b)
# Dequantize
xh = dequantizer(symb_index, b)
# Quantization error
q_error = np.max(np.abs(cs - xh))
print("Max Quantization Error", q_error)