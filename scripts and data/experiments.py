import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis
from matplotlib import pyplot as plt
from scipy.io import wavfile
from dct import iframeDCT, frameDCT
from subband import codec0
from psychoacoustics import DCTpower, Dksparse, STinit
from barks import frequency_in_barks

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

plt.figure(figsize=(15,4.8))
plt.xticks(np.arange(sr, step=4000))
for i in range(0,M):
    H_f = np.fft.fft(H[:, i])
    H_f = 10 * np.log10(np.square(np.abs(H_f)))
    f = np.arange(0, sr / 2, sr / len(H_f))
    plt.plot(f,H_f[:len(f)])
plt.xlabel('Συχνότητα f (Hz)') #Τίτλος στον άξονα x
plt.ylabel('Μέτρο (dB)') #Τίτλος στον άξονα y
plt.title('Μέτρο συναρτήσεων μεταφοράς') #Τίτλος του διαγράμματος
plt.show()

#Plot frequency in barks diagram
plt.figure(figsize=(15,4.8))
for i in range(0,M):
    H_f = np.fft.fft(H[:, i])
    H_f = 10 * np.log10(np.square(np.abs(H_f)))
    f = np.arange(0, sr / 2, sr / len(H_f))
    f_in_barks = frequency_in_barks(f)
    plt.plot(f_in_barks, H_f[:len(f_in_barks)])

plt.xlabel('Συχνότητα f (barks)') #Τίτλος στον άξονα x
plt.ylabel('Μέτρο (dB)') #Τίτλος στον άξονα y
plt.title('Μέτρο συναρτήσεων μεταφοράς') #Τίτλος του διαγράμματος
plt.show()

sr, x_data = wavfile.read('myfile.wav')
x_hat, Y_tot = codec0('myfile.wav',h, M,N)

#Πειράματα για DCT
c = frameDCT(Y_tot)
Y_tot_hat = iframeDCT(c)

#Πειράματα Μέρους 3 - Psychoacoustics

P = DCTpower(c)
size = M *N
D = Dksparse(size)
print(D)
ST = STinit(c,D)