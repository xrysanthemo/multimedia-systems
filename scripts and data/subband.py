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


#opening myfile.wav
sr, data = wavfile.read('myfile.wav')
#sampling interval
ts = 1.0/sr
Y = frame_sub_analysis(data, H, N) #xbuff is the input signal
Z = frame_sub_synthesis(Y, G)


# def get_frame(h, N):
#     Y = frame_sub_analysis(data, h, N)
#     Z = frame_sub_synthesis(Y, G) #Z is the frame
#     return Z
#na thumithw giati avta
 #giati?/
l = np.arange(L)
T = L/sr
f = l / T

plt.figure(figsize=(15,4.8))
plt.xticks(np.arange(sr, step=4000))
for i in range(0,M):
    H_f = np.fft.fft(H[:, i])
    H_f = 10 * np.log10(np.square(np.abs(H_f)))
    plt.plot(f, H_f)
plt.xlabel('Συχνότητα f (Hz)') #Τίτλος στον άξονα x
plt.ylabel('Μέτρο (dB)') #Τίτλος στον άξονα y
plt.title('Μέτρο συναρτήσεων μεταφοράς (Hz)') #Τίτλος του διαγράμματος
plt.show()

#TODO: Plot function, argument:scale, returning db
#TODO: Implement a good colormapping
#TODO: Check if plot is right to plot 2 highs in each filter
#TODO:Automatically define figsize
def frequency_in_barks(f): #thn pairnw ws np array, vlepoume
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.square(np.arctan(f / 7500))

#Plot frequency in barks diagram
plt.figure(figsize=(15,4.8))
f_in_barks = frequency_in_barks(f)
for i in range(0,M):
    H_f = np.fft.fft(H[:, i])
    H_f = 10 * np.log10(np.square(np.abs(H_f)))
    plt.plot(f_in_barks, H_f)
plt.xlabel('Συχνότητα f (barks)') #Τίτλος στον άξονα x
plt.ylabel('Μέτρο (dB)') #Τίτλος στον άξονα y
plt.title('Μέτρο συναρτήσεων μεταφοράς (Hz)') #Τίτλος του διαγράμματος
plt.show()


#4th task
def codec0(wavin, h, M, N):
    #Διαβάζω το αρχείο .wav
    sr, data = wavfile.read(wavin)
    #Κατασκευάζω τα φίλτρα ανάλυσης και σύνθεσης
    H = make_mp3_analysisfb(h, M)
    G = make_mp3_synthesisfb(h, M)
    #Δημιουργώ το xhat
    xhat = []
    # sampling interval
    ts = 1.0 / sr
    L = len(h) #512, M=21 N=36
    Y_tot = np.zeros([len(data), M])
    #Y_tot = []
    lower = 0
    for i in range(0,math.ceil(len(data))):
        # upper = (i + 1) * (M * (N-1) + L)
        upper = (i+1) * M * N
        samples = data[lower:upper]
        if lower + ((N-1)*M + L)  < len(data):
            buffer = data[lower: lower + ((N-1)*M + L)]
        else:
            buffer = data[lower:len(data)]
        lower = upper + 1
        #Frame Y mesw analysis se bands
        Y = frame_sub_analysis(buffer, H, N)
        Yc = donothing(Y)
        bound1 = i * N
        bound2 = (i +1) * N
        Y_tot[bound1:bound2, :] = Yc
        Yh = idonothing(Yc)
        Z = frame_sub_synthesis(Yh, G)
        xhat.append(Z)
    #return xhat, Y_tot
    #TODO: na vrw megethos tou Ytot
    return Y_tot
y = codec0('myfile.wav', h, M, N)
print(y)