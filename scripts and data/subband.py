import math
import numpy as np
from numpy import average
from math import log10
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis
from matplotlib import pyplot as plt
from scipy.io import wavfile
from nothing import donothing, idonothing
from scipy import stats
from dct import frameDCT, iframeDCT
from psychoacoustics import DCTpower

def get_impulse_response():
    # read numpy file
    h = np.load('h.npy', allow_pickle=True).item()
    h_coefficients = h["h"]
    return h_coefficients

#Define the number of subbands
M = 32 #num of filters
L = 512 #len of filters
N = 36 #num of samples
h = get_impulse_response().reshape(512,)
H = make_mp3_analysisfb(h, M)
G = make_mp3_synthesisfb(h, M)

#4th task
def codec0(wavin, h, M, N):
    #Διαβάζω το αρχείο .wav
    sr, data = wavfile.read(wavin) #514944
    #Κατασκευάζω τα φίλτρα ανάλυσης και σύνθεσης
    H = make_mp3_analysisfb(h, M)
    G = make_mp3_synthesisfb(h, M)
    #Δημιουργώ την παράμετρο για το l
    L = len(h)  # 512
    # Μέγεθος buffer
    xbuffer_size = (N - 1) * M + L
    ybuffer_rows = int((N - 1) + L / M)

    # Ορίζω το offset του buffer
    xoffset = xbuffer_size - M * N
    yoffset = ybuffer_rows - N

    #Υλοποιώ padding του σήματός μου για να μπορέσει να αναλυθεί σε subband
    padding = np.zeros((L - M,))
    iters = math.ceil(len(data) / (M * N))
    # Αρχικοποιώ Y_tot, xhat
    Y_tot = np.zeros((N * iters, M))
    xhat = np.zeros(data.shape)
    xbuff = np.zeros([xbuffer_size])
    ybuff = np.zeros([ybuffer_rows, M])
    # data = np.append(data,padding)

    for i in range(0,iters): #to kanw mia fora tha to kanw expand meta
    #Διαβάζω το buffer: (N − 1)M + L
    #Διαβάζω τα δείγματά μου: M*N
        # original_samples = data[(i*M*N):(i+1)*M*N]
        # buffer_samples = data[(i*M*N):(i*M*N) + xbuffer_size]
        xbuff[xoffset:xbuffer_size] = data[i * M * N:(i + 1) * M * N]

    #Frame Sub Analysis sto buffer mou
        Y = frame_sub_analysis(xbuff, H, N)
    # Shift xbuffer
        xbuff[0:xoffset] = xbuff[xoffset:2 * xoffset]

    #Επεξεργασία του frame
        Yc = donothing(Y)
    #Συσσώρευση
        bound1 = i * N
        bound2 = (i + 1) * N
        Y_tot[bound1:bound2, :] = Yc
    #Αντιστροφή της διαδικασίας
        Yh = idonothing(Yc)

        ybuff[yoffset:ybuffer_rows, :] = Yh
    #Παραγωγή δειγμάτων synthesis
        Z = frame_sub_synthesis(ybuff, G)
    # Shift ybuffer
        ybuff[0:yoffset, :] = ybuff[yoffset:2 * yoffset, :]
    #Συσσώρευση σε xhat
        xhat[(bound1 *M):(bound2*M)] = Z
    err = data - xhat
    plt.plot(err)
    plt.show()
    #Write file to another file in our folder
    wavfile.write("MYFILE_CODECO.wav", sr, xhat.astype(np.int16))
    return xhat.astype(np.int16), Y_tot
sr, x_data = wavfile.read('myfile.wav')
x_hat, Y_tot = codec0('myfile.wav',h, M,N)
print(Y_tot)
def signalPower(x):
    return np.mean(np.square(x,dtype='int64')) #to prevent overflow


def SNRsystem(inputSig, outputSig):
    noise = outputSig - inputSig
    powS = signalPower(outputSig)
    powN = signalPower(noise)
    return powS/powN

#Experiments - SNR
SNR = SNRsystem(x_data, x_hat)
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(x_data[0:5000])
axs[1].plot(x_hat[0:5000])
plt.show()

plt.plot(x_data[0:5000])
plt.plot(x_hat[0:5000])
plt.title("Comparison between the two bitstreams")
plt.show()

# print(SNR)


#Coder only
def coder0(wavin, h,M,N):
    # Διαβάζω το αρχείο .wav
    sr, data = wavfile.read(wavin)  # 514944
    # Κατασκευάζω τα φίλτρα ανάλυσης
    H = make_mp3_analysisfb(h, M)
    # Δημιουργώ την παράμετρο για το l
    L = len(h)  # 512
    buffer_size = (N - 1) * M + L
    # Υλοποιώ padding του σήματός μου για να μπορέσει να αναλυθεί σε subband
    padding = np.zeros((L - M,))
    iters = math.ceil(len(data) / (M * N))
    data = np.append(data, padding)
    # Αρχικοποιώ Y_tot, xhat
    Y_tot = np.zeros((N * iters, M))
    for i in range(0, iters):  # to kanw mia fora tha to kanw expand meta
        # Διαβάζω το buffer: (N − 1)M + L
        # Διαβάζω τα δείγματά μου: M*N
        original_samples = data[(i * M * N):(i + 1) * M * N]
        buffer_samples = data[(i * M * N):(i * M * N) + buffer_size]
        # Frame Sub Analysis sto buffer mou
        Y = frame_sub_analysis(buffer_samples, H, N)
        # Επεξεργασία του frame
        Yc = donothing(Y)
        # Συσσώρευση
        bound1 = i * N
        bound2 = (i + 1) * N
        Y_tot[bound1:bound2, :] = Yc
    return Y_tot

#Decoder Implementation
def decoder0(Y_tot, h,M,N):
    xhat = np.zeros(data.shape) #??????
    return xhat

