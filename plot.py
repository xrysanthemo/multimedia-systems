
import numpy as np
import matplotlib.pyplot as plt
from hz2barks import Hz2Barks

def plot_H_Hz(H):
    plt.figure(1)

    M = H.shape[1]

    # Frequency axis
    freq = np.linspace(-22050, 22050, 512)

    # Noise floor to avoid division by zero
    noise_floor = 1e-10

    # Plot the transfer functions of each column on a common graph
    for i in range(M):
        H_col = H[:, i]
        H_col_mag = 10 * np.log10(np.abs(H_col) ** 2 + noise_floor)
        plt.plot(freq, H_col_mag, label=f'H filter: {i + 1}')


    # Add labels and legend
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()

    # Show the plot
    plt.show()


def plot_H_barks(H):
    plt.figure(2)

    M = H.shape[1]

    # Frequency axis
    freq = np.linspace(-22050, 22050, 512)
    # Convert frequency to the Bark scale
    barks = Hz2Barks(freq)

    # Noise floor to avoid division by zero
    noise_floor = 1e-10

    # Plot the transfer functions of each column on a common graph
    for i in range(M):
        H_col = H[:, i]
        H_col_mag = 10 * np.log10(np.abs(H_col) ** 2 + noise_floor)
        plt.plot(barks, H_col_mag, label=f'H filter: {i + 1}')

    # Add labels and legend
    plt.xlabel('Barks')
    plt.ylabel('Magnitude (dB)')
    plt.legend()

    # Show the plot
    plt.show()