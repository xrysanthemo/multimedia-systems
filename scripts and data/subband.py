import math
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis
from scipy.io import wavfile
from nothing import donothing, idonothing
from plot import plot_H_Hz, plot_H_barks, plot_err, plot_snr
from matplotlib import pyplot as plt

def get_impulse_response():
    # read numpy file
    h = np.load('h.npy', allow_pickle=True).item()
    h_coefficients = h["h"]
    return h_coefficients

def codec0(wavin, h, M, N):
    #Διαβάζω το αρχείο .wav
    sr, data = wavfile.read(wavin) #514944
    #Κατασκευάζω τα φίλτρα ανάλυσης και σύνθεσης
    H = make_mp3_analysisfb(h, M)
    G = make_mp3_synthesisfb(h, M)
    #Δημιουργώ την παράμετρο για το l
    L = len(h)  # 512
    MN = M * N
    data_len = len(data)
    # Μέγεθος buffer
    xbuffer_size = (N - 1) * M + L
    ybuffer_rows = int((N-1) + L / M)
    # Buffers
    xbuff = np.zeros([xbuffer_size])
    ybuff = np.zeros([ybuffer_rows, M])
    # Ορίζω το offset του buffer
    xoffset = xbuffer_size - MN
    yoffset = ybuffer_rows - N

    iters = math.ceil(data_len / MN)

    # Αρχικοποιώ Y_tot, xhat
    Y_tot = np.zeros((N * iters, M))
    xhat = np.zeros(data.shape)

    for i in range(0, iters):
    # Fill buffer
        xbuff[xoffset:xbuffer_size] = data[i * MN:(i + 1) * MN]
        # plt.plot(xbuff, 'g')
        # plt.plot(data[i * MN:(i + 1) * MN], 'r')
        # plt.title("Without Shift")
        # plt.show()
    # Frame Sub Analysis sto buffer mou
        Y = frame_sub_analysis(xbuff, H, N)
    # Shift xbuffer
        xbuff[0:xoffset] = xbuff[xbuffer_size - xoffset:]
        # plt.plot(xbuff, 'g')
        # plt.plot(data[i * MN:(i + 1) * MN], 'r')
        # plt.title("With Shift")
        # plt.show()
    # Επεξεργασία του frame
        Yc = donothing(Y)
    # Συσσώρευση
        bound1 = i * N
        bound2 = (i + 1) * N
        Y_tot[bound1:bound2, :] = Yc
    # Αντιστροφή της διαδικασίας
        Yh = idonothing(Yc)
        ybuff[yoffset:ybuffer_rows, :] = Yh
    # Παραγωγή δειγμάτων synthesis
        Z = frame_sub_synthesis(ybuff, G)
    # Shift ybuffer
        ybuff[0:yoffset, :] = ybuff[ybuffer_rows - yoffset:, :]
    # Συσσώρευση σε xhat
        xhat[(bound1*M):(bound2*M)] = Z

    # Write file to another file in our folder
    #ena teleutaio shift sto xhat
    val = xhat[0:xoffset]
    xhat[0:(len(xhat) - xoffset)] = xhat[xoffset:]
    xhat[(len(xhat) - xoffset):] = val

    wavfile.write("MYFILE_CODECO.wav", sr, xhat.astype(np.int16))
    return xhat.astype(np.int16), Y_tot

def SNRsystem(inputSig, outputSig):
    noise = np.float64(outputSig - inputSig)
    powS = np.mean(np.float64(outputSig)**2)
    powN = np.mean(noise**2)
    return 10*np.log10(powS/powN)

#Coder Implementation
def coder0(wavin, h,M,N):
    # Διαβάζω το αρχείο .wav
    sr, data = wavfile.read(wavin)  # 514944
    # Κατασκευάζω το φίλτρο ανάλυσης
    H = make_mp3_analysisfb(h, M)
    L = len(h)  # 512
    MN = M * N

    # Μέγεθος buffer
    xbuffer_size = (N - 1) * M + L
    # Buffers
    xbuff = np.zeros([xbuffer_size])
    # Ορίζω το offset του buffer
    xoffset = xbuffer_size - MN

    iters = math.ceil(len(data) / (MN))

    # Αρχικοποιώ Y_tot, xhat
    Y_tot = np.zeros((N * iters, M))

    for i in range(0, iters):
        # Fill buffer
        xbuff[xoffset:xbuffer_size] = data[i * MN:(i + 1) * MN]
        # Frame Sub Analysis sto buffer mou
        Y = frame_sub_analysis(xbuff, H, N)
        # Shift xbuffer
        xbuff[0:xoffset] = xbuff[xoffset:2 * xoffset]
        # Επεξεργασία του frame
        Yc = donothing(Y)
        # Συσσώρευση
        bound1 = i * N
        bound2 = (i + 1) * N
        Y_tot[bound1:bound2, :] = Yc
    return Y_tot

#Decoder Implementation
def decoder0(Y_tot, h, M, N):
    sr = 44100
    data_len = Y_tot.shape[0] * Y_tot.shape[1]
    # Κατασκευάζω το φίλτρο σύνθεσης
    G = make_mp3_synthesisfb(h, M)
    L = len(h)  # 512
    MN = M * N

    # Μέγεθος buffer
    ybuffer_rows = int((N - 1) + L / M)
    # Buffers
    ybuff = np.zeros([ybuffer_rows, M])
    # Ορίζω το offset του buffer
    yoffset = ybuffer_rows - N

    iters = math.ceil(data_len / (MN))
    xhat = np.zeros(data_len)

    for i in range(0, iters):
        bound1 = i * N
        bound2 = (i + 1) * N
        Yc = Y_tot[bound1:bound2, :]
        # Αντιστροφή της διαδικασίας
        Yh = idonothing(Yc)
        ybuff[yoffset:ybuffer_rows, :] = Yh
        # Παραγωγή δειγμάτων synthesis
        Z = frame_sub_synthesis(ybuff, G)
        # Shift ybuffer
        ybuff[0:yoffset, :] = ybuff[yoffset:2 * yoffset, :]
        # Συσσώρευση σε xhat
        xhat[(bound1 * M):(bound2 * M)] = Z

    # Write file to another file in our folder
    wavfile.write("MYFILE_DECODER.wav", sr, xhat.astype(np.int16))
    return xhat.astype(np.int16)

