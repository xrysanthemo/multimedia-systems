import numpy as np
import matplotlib.pyplot as plt
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from scipy.io import wavfile
from dct import iframeDCT, frameDCT
from subband import codec0, get_impulse_response, SNRsystem, coder0, decoder0
from psychoacoustics import DCTpower, Dksparse, STinit, MaskPower, get_hearing_threshold, STreduction, Hz2Barks, psycho
from quantization import critical_bands, DCT_band_scale, quantizer, dequantizer, all_bands_quantizer,  all_bands_dequantizer
from rle import RLE, iRLE
from huffdelo import huff, ihuff
from file_handler import write_huff, read_huff, create_huff
from assembled_mp3 import MP3codec, MP3decod, MP3cod
from plot import plot_H_Hz, plot_H_barks, plot_stream_comparison, plot_err

# Define Parameters
M = 32 #αριθμός filters
L = 512 #αριθμός filters
N = 36 #αριθμός samples
MN = M * N #μήκος frame

#συντελεστές mother wavelet για τη διαδικασία analysis/synthesis
h = get_impulse_response().reshape(512,)
#Διαδικασία για τους πίνακες make_mp3_analysis και make_mp3_synthesis
H = make_mp3_analysisfb(h, M)
G = make_mp3_synthesisfb(h, M)

#Ανάγνωση του .wav αρχείου
sr, data = wavfile.read('myfile.wav')

# # Subband Filtering 3.1
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Subband Filtering 3.1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
#Print frame sub analysis and sub synthesis arrays
print("Frame Sub Analysis: ", H, "\n")
print("Frame Sub Synthesis: ", G,"\n")
#Plot H Hz - barks
plot_H_Hz(H, sr)
plot_H_barks(H, sr)

# Codec
x_hat, Y_tot = codec0('myfile.wav', h, M, N)
plot_err(data, x_hat)
plot_stream_comparison(data, x_hat)

# Coder - Decoder
Y_tot2 = coder0('myfile.wav', h, M, N)
x_hat2 = decoder0(Y_tot2, h, M, N)

print("x_hat diff: ", np.mean(np.mean(x_hat - x_hat2)))
print("Y_tot diff: ", np.mean(np.mean(Y_tot - Y_tot2)))
plot_err(data, x_hat2)
plot_stream_comparison(data, x_hat2)

#Experiments - SNR
SNR = SNRsystem(data, x_hat)
print("SNR codec0: ", SNR, " dB")
SNR = SNRsystem(data, x_hat2)
print("SNR coder0 - decoder0: ", SNR, " dB")

# DCT 3.2
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DCT 3.2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
dummy_frame = Y_tot[36*10:36*11]
c = frameDCT(Y_tot)
c_of_frame = frameDCT(dummy_frame)
dummy_frame_hat = iframeDCT(c_of_frame)
print("Difference between original frame and inverse DCT: ", np.mean(np.mean(dummy_frame - dummy_frame_hat)))

# Psychoacoustics 3.3
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Psychoacoustics 3.3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
# Υπολογισμος συχνοτικών περιοχών
Tq = get_hearing_threshold()
plt.plot(Tq[0, :])
plt.title("Hearing Threshold in quietness")
plt.show()
D = Dksparse(MN)
# Υπολογισμός κατωφλίου ακουστότητας
Tg = psycho(c_of_frame, D)
plt.plot(Tg)
plt.title("Hearing Threshold after maskers' influence")
plt.show()

# Quantization 3.4
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Quantization 3.4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
# Scale DCT
cs, sc = DCT_band_scale(c_of_frame)
# bits num
x = np.linspace(-1,1, 1152)
for b in range(1, 6):
    # Quantize
    symb_index = quantizer(cs, b)
    # Dequantize
    ch = dequantizer(symb_index, b)
    # Quantization error
    q_error = np.max(np.abs(cs - ch))
    print("Max Quantization Error with ", b, " bits: ", q_error)

    #To just plot the quantizer
    symb_index = quantizer(x, b)
    # Dequantize
    xh = dequantizer(symb_index, b)
    plt.plot(x, xh)
    title = "Quantizer-Dequantizer: " + str(b) + " bits"
    plt.title(title)
    plt.show()

# RLE 3.5
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RLE 3.5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

#Example RLE/IRLE for some dummy symbol indexes
symb_index_dummy1 = [0, 0, 3, 4, 9, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, -2]
symb_index_dummy2 = [3, 4, 9, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, -2]
symb_index_dummy3 = [0]
symb_index_dummy4 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

rle1 = RLE(symb_index_dummy1, len(symb_index_dummy1))
rle2 = RLE(symb_index_dummy2,len(symb_index_dummy2))
rle3 = RLE(symb_index_dummy3,len(symb_index_dummy3))
rle4 = RLE(symb_index_dummy4,len(symb_index_dummy4))

inverse_dummy_1 = iRLE(rle1, len(symb_index_dummy1))
inverse_dummy_2 = iRLE(rle2, len(symb_index_dummy2))
inverse_dummy_3 = iRLE(rle3, len(symb_index_dummy3))
inverse_dummy_4 = iRLE(rle4, len(symb_index_dummy4))
#Checking if inverse is returning the correct symbol indexes, error should be 0
print("RLE - iRLE Error's sum: ", sum(inverse_dummy_1 - symb_index_dummy1))
print("RLE - iRLE Error's sum: ", sum(inverse_dummy_2 - symb_index_dummy2))
print("RLE - iRLE Error's sum: ", sum(inverse_dummy_3 - symb_index_dummy3))
print("RLE - iRLE Error's sum: ", sum(inverse_dummy_4 - symb_index_dummy4))

# Huffman 3.6
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Huffman 3.6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
datalen = len(data)
all_c = []
create_huff("dummy_huffman.txt")
#Example for some frames
for i in range(350,353):
    symb_index_r, SF, B = all_bands_quantizer(c[MN*i:MN*(i+1)], Tg)
    print("\n---- frame: ", i, " bits: ", B, " ----\n")
    ch = all_bands_dequantizer(symb_index_r, B, SF)

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

    run_symbols_rle = RLE(symb_index_r_flat, MN - 1)
    symb_index_rle = iRLE(run_symbols_rle, MN - 1)
    print("Before RLE: ", symb_index_r_flat, "\nAfter RLE:\n", run_symbols_rle)

    frame_stream, frame_symbol_prob = huff(run_symbols_rle)
    run_symbols_huff = ihuff(frame_stream, frame_symbol_prob)
    write_huff("dummy_huffman.txt", frame_stream)
    error = run_symbols_huff - run_symbols_rle
    print("Huffman error sum: ", sum(sum(error)))

print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MP3 Codec ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
x_hatMP3, Y_totMP3 = MP3codec('myfile.wav', h, M, N)
plot_err(data, x_hatMP3)
plot_stream_comparison(data, x_hatMP3)

# # Coder - Decoder
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MP3 Coder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
Y_totMP3_2 = MP3cod('myfile.wav', h, M, N)
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MP3 Decoder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
x_hatMP3_2 = MP3decod(Y_totMP3_2, h, M, N)

print("x_hat diff: ", np.mean(np.mean(x_hatMP3 - x_hatMP3_2)))
plot_err(data, x_hatMP3_2)
plot_stream_comparison(data, x_hatMP3_2)

#Experiments - SNR
SNR_MP3 = SNRsystem(data, x_hatMP3)
print("SNR codec0: ", SNR_MP3, " dB")
SNR_MP3 = SNRsystem(data, x_hatMP3_2)
print("SNR coder0 - decoder0: ", SNR_MP3, " dB")