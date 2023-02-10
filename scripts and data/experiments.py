import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from scipy.io import wavfile
from dct import iframeDCT, frameDCT
from subband import codec0, get_impulse_response, SNRsystem, coder0, decoder0
from psychoacoustics import DCTpower, Dksparse, STinit, MaskPower, get_hearing_threshold, STreduction, Hz2Barks, psycho
from quantization import critical_bands, DCT_band_scale, quantizer, dequantizer, all_bands_quantizer,  all_bands_dequantizer
from rle import RLE, iRLE
from huffdelo import huff, ihuff
from file_handler import write_huff, read_huff, create_huff
from assembled_mp3 import MP3codec

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
# x_hat, Y_tot = codec0('myfile.wav', h, M, N)

x_hat, Y_tot = MP3codec('myfile.wav', h, M, N)

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

# # Πειράματα Quantization
# # Scale DCT
# cs, sc = DCT_band_scale(c[MN*0:MN*1])
# # bits num
# b = 3
# # Quantize
# symb_index = quantizer(cs, b)
# # Dequantize
# xh = dequantizer(symb_index, b)
# # Quantization error
# q_error = np.max(np.abs(cs - xh))
# print("Max Quantization Error", q_error)

datalen = len(data)
all_c = []
create_huff("huffman.txt")
for i in range(358,359):
    symb_index_r, SF, B = all_bands_quantizer(c[MN*i:MN*(i+1)], Tg)
    # print("frame: ", i, " bits: ", B)
    ch = all_bands_dequantizer(symb_index_r, B, SF)
    all_c.append(ch)

    # FLATTENING
    symb_index_r_flat = [int(item) for sublist in symb_index_r for item in sublist]

    # UNFLATTENING
    cb = critical_bands(MN)
    symb_index_sublist = []
    symb_index_unflat = []
    current_index = 0
    for i in range(1, int(max(cb)+1)):
        count = np.count_nonzero(cb == i)
        symb_index_sublist = np.asarray(symb_index_r_flat[current_index:current_index + count])
        symb_index_unflat.append(symb_index_sublist)
        current_index = current_index + count
    # END OF UNFLATTENING

    run_symbols_rle = RLE(symb_index_r_flat, len(symb_index_r_flat))
    symb_index_rle = iRLE(run_symbols_rle, len(symb_index_r_flat))

    frame_stream, frame_symbol_prob = huff(run_symbols_rle)
    run_symbols_huff = ihuff(frame_stream, frame_symbol_prob)
    write_huff("huffman.txt", frame_stream)
    file = read_huff("huffman.txt")
    # print(all(symb_index_rle == symb_index_r_flat))
    # error = run_symbols_huff - run_symbols_rle

all_c = [item for sublist in all_c for item in sublist]


#RLE

# symb_index_dummy1 = [0, 0, 3, 4, 9, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, -2]
# symb_index_dummy2 = [3, 4, 9, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, -2]
# symb_index_dummy3 = [0]
# symb_index_dummy4 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#
# symb_index_r, SF, B = all_bands_quantizer(c[MN*2:MN*3], Tg)
# symb_index_r_flattened = [item for sublist in symb_index_r for item in sublist]
#
# rle = RLE(symb_index_dummy1, len(symb_index_dummy1))
# rle2 = RLE(symb_index_dummy2,len(symb_index_dummy2))
# rle3 = RLE(symb_index_dummy3,len(symb_index_dummy3))
# rle4 = RLE(symb_index_dummy4,len(symb_index_dummy4))
#
# inverse_dummy_1 = iRLE(rle, len(symb_index_dummy1))
# inverse_dummy_2 = iRLE(rle2, len(symb_index_dummy2))
# inverse_dummy_3 = iRLE(rle3, len(symb_index_dummy3))
# inverse_dummy_4 = iRLE(rle4, len(symb_index_dummy4))
#
# print(inverse_dummy_1 - symb_index_dummy1)
# print(inverse_dummy_2 - symb_index_dummy2)
# print(inverse_dummy_3 - symb_index_dummy3)
# print(inverse_dummy_4 - symb_index_dummy4)





