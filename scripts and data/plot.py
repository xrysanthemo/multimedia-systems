import numpy as np
import matplotlib.pyplot as plt
from psychoacoustics import Hz2Barks

#Plot H in Hz diagram
def plot_H_Hz(H:np.ndarray, sr:int):
    """
    Απεικόνιση της συνάρτησης μεταφοράς H στην κλίμακα Hz
    """
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
def plot_H_barks(H:np.ndarray, sr:int):
    """
    Απεικόνιση της συνάρτησης μεταφοράς H στην κλίμακα Barks
    """
    M = H.shape[1]
    plt.figure(figsize=(15,4.8))

    for i in range(0,M):
        H_f = np.fft.fft(H[:, i])
        H_f = 10 * np.log10(np.square(np.abs(H_f)))
        f = np.arange(0, sr / 2, sr / len(H_f))
        f_in_barks = Hz2Barks(f)
        plt.plot(f_in_barks, H_f[:len(f_in_barks)], label=f'H filter: {i + 1}')
    plt.xlabel('Συχνότητα z (barks)') #Τίτλος στον άξονα x
    plt.ylabel('Μέτρο (dB)') #Τίτλος στον άξονα y
    plt.legend()
    plt.title('Μέτρο συναρτήσεων μεταφοράς') #Τίτλος του διαγράμματος
    plt.show()

#Plot SNR
def plot_stream_comparison(x_data:np.ndarray, x_hat:np.ndarray):
    """
    Εμφανίζει σε κοινό Plot το αποκωδικοποιημένο και το αρχικό stream για σύγκριση
    Για καλή οπτικοποίηση επιλέγουμε να εμφανίσουμε μόνο ένα slice από τα δεδομένα
    """
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x_data[0:2000])
    axs[0].set_title('Indicative Samples of the Original Stream')
    axs[1].plot(x_hat[0:2000])
    axs[1].set_title('Indicative Samples of the Decoded Stream')
    plt.show()
    plt.plot(x_data)
    plt.plot(x_hat)
    plt.title("Comparison between the two bitstreams")
    plt.show()

# Plot error data - xhat
def plot_err(data:np.ndarray, x_hat:np.ndarray):
    """
    Απεικόνιση της διαφοράς μεταξύ κωδικοποιημένου και αποκωδικοποιημένου stream
    """
    plt.plot(data - x_hat)
    plt.title("Error between data and x_hat")
    plt.show()