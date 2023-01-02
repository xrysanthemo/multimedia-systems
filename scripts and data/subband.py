import math
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis
from matplotlib import pyplot as plt
from scipy.io import wavfile
from nothing import donothing, idonothing



def get_impulse_response():
    #read numpy file
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
    sr, data = wavfile.read(wavin)
    #Κατασκευάζω τα φίλτρα ανάλυσης και σύνθεσης
    H = make_mp3_analysisfb(h, M)
    G = make_mp3_synthesisfb(h, M)
    #Δημιουργώ την παράμετρο για το l
    L = len(h)  # 512
    iters = math.ceil(len(data)/(M*N))
    #Αρχικοποιώ Y_tot, xhat
    Y_tot = np.zeros((N*iters , M))
    xhat = np.zeros((N*M*iters, ))
    for i in range(0,446): #to kanw mia fora tha to kanw expand meta
    #Διαβάζω το buffer: (N − 1)M + L
        buffer_size =  (N-1) * M +L
        buffer_samples = data[(i * buffer_size):((i + 1) * buffer_size)]
    #Διαβάζω τα δείγματά μου: M*N
        original_samples = data[(i*M*N):(i+1)*M*N]
        buffer_samples = data[(i*M*N):(i*M*N)+buffer_size]
    #Frame Sub Analysis sto buffer mou
        Y = frame_sub_analysis(buffer_samples, H, N)
    #Επεξεργασία του frame
        Yc = donothing(Y)
    #Συσσώρευση
        bound1 = i * N
        bound2 = (i + 1) * N
        Y_tot[bound1:bound2, :] = Yc
    #Αντιστροφή της διαδικασίας
        Yh = idonothing(Yc)
    #Παραγωγή δειγμάτων synthesis
        Z = frame_sub_synthesis(Yh, G)
    #Συσσώρευση σε xhat
        xhat[(bound1 *M):(bound2*M)] = Z
    return xhat, Y_tot

x, y = codec0('myfile.wav',h, M,N)
print(y)