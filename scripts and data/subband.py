
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis
from matplotlib import pyplot as plt
from scipy.io import wavfile
from cycler import cycler


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

#Transfer function creation

# for i in range(0,M):
#     H_f = np.fft.fft(H[:,i])
#     print(H_f)

#opening myfile.wav
samplerate, data = wavfile.read('myfile.wav')
Y = frame_sub_analysis(data, H, N) #xbuff is the input signal
Z = frame_sub_synthesis(Y, G)
print(Z)

def get_frame(h, N):
    Y = frame_sub_analysis(data, h, N)
    Z = frame_sub_synthesis(Y, G) #Z is the frame
    return Z

freq = np.arange(0, samplerate, samplerate/512)
for i in range(0,M):
    H_f = np.fft.fft(H[:, i])
    H_f = 10 * np.log10(np.square(np.abs(H_f)))
    plt.plot(freq, H_f)


plt.xlabel('Συχνότητα f (Hz)') #Τίτλος στον άξονα x
plt.ylabel('Μέτρο (dB)') #Τίτλος στον άξονα y
plt.title('Μέτρο συναρτήσεων μεταφοράς (Hz)') #Τίτλος του διαγράμματος
plt.show()

def frequency_in_barks(freq): #thn pairnw ws np array, vlepoume
    return 13 * np.arctan(0.00076 * freq) + 3.5 * np.square(np.arctan(freq/7500))
