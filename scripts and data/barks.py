import numpy as np
def frequency_in_barks(f): #thn pairnw ws np array, vlepoume
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.square(np.arctan(f / 7500))