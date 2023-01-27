
import numpy as np
import matplotlib.pyplot as plt
from psychoacoustics import Hz2Barks

#Plot H in Hz diagram
def plot_H_Hz(H, sr):
    M = H.shape[1]
    plt.figure(figsize=(15, 4.8))
    plt.xticks(np.arange(sr, step=4000))
    for i in range(0,M):
        H_f = np.fft.fft(H[:, i])
        H_f = 10 * np.log10(np.square(np.abs(H_f)))
        f = np.arange(0, sr / 2, sr / len(H_f))
        plt.plot(f,H_f[:len(f)], label=f'H filter: {i + 1}')
    plt.xlabel('Συχνότητα f (Hz)') #Τίτλος στον άξονα x
    plt.ylabel('Μέτρο (dB)') #Τίτλος στον άξονα y
    plt.legend()
    plt.title('Μέτρο συναρτήσεων μεταφοράς') #Τίτλος του διαγράμματος
    plt.show()

#Plot H in barks diagram
def plot_H_barks(H, sr):
    M = H.shape[1]
    plt.figure(figsize=(15,4.8))
    for i in range(0,M):
        H_f = np.fft.fft(H[:, i])
        H_f = 10 * np.log10(np.square(np.abs(H_f)))
        f = np.arange(0, sr / 2, sr / len(H_f))
        f_in_barks = Hz2Barks(f)
        plt.plot(f_in_barks, H_f[:len(f_in_barks)], label=f'H filter: {i + 1}')
    plt.xlabel('Συχνότητα f (barks)') #Τίτλος στον άξονα x
    plt.ylabel('Μέτρο (dB)') #Τίτλος στον άξονα y
    plt.legend()
    plt.title('Μέτρο συναρτήσεων μεταφοράς') #Τίτλος του διαγράμματος
    plt.show()

#Plot SNR
def plot_snr(x_data, x_hat):
    x_len = len(x_hat)
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x_data[0:2000])
    axs[1].plot(x_hat[0:2000])
    plt.show()
    plt.plot(x_data[0:2000])
    plt.plot(x_hat[0:2000])
    plt.title("Comparison between the two bitstreams")
    plt.show()

# Plot error data - xhat
def plot_err(data, x_hat):
    plt.plot(data - x_hat)
    plt.title("Error between data and x_hat")
    plt.show()