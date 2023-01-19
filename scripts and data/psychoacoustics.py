import numpy as np
from scipy.sparse import csr_matrix

def Hz2Barks(f): #thn pairnw ws np array, vlepoume
    return np.asarray(13 * np.arctan(0.00076 * f) + 3.5 * np.arctan(np.square(f / 7500)))

def DCTpower(c):
    #ypologizei isxy se db apo suntelestes dct symfwna me auth th sxesh
    P = 10 * np.log10(np.square(abs(c)))
    return P

def Dksparse(Kmax): #Η είσοδος Kmax αντιστοιχεί στη μέγιστη διακριτη συχνότητα (εδώ Kmax = MN − 1)
    row = []
    col = []
    data =[]
    #k diakrites zwnes - grammes
    for k in range(0, Kmax): #sixnotita j e Dk - stiles
        if 2 < k and k < 282:
            col.append(k)
            row.append(2)
            data.append(1)
        elif 282 <= k and k < 570:
            for ind in range(2, 14):
                col.append(k)
                row.append(ind)
                data.append(1)
        elif 570 <= k and k < 1152:
            for ind in range(2, 28):
                col.append(k)
                row.append(ind)
                data.append(1)
    D = csr_matrix((data, (row, col)), shape=(Kmax, Kmax)) #.toarray()
    return D #έξοδος είναι sparse matrix Kmax * Kmax

def STinit(c,D):
    ST = []
    P = DCTpower(c)
    Kmax = D.shape[0]
    for k in range(1, Kmax - 1):
        (rows, cols) = D[:, k].nonzero()
        if (P[k] > P [k + 1] and P[k] > P[k-1]):
            for i in rows:
                if (k+i < 1152 and P[k] > P[k + i] + 7 and P[k] > P[k - i] + 7):
                    ST.append(k)
    ST = sorted(list(set(ST)))
    return ST

def MaskPower(c, ST):
    P = DCTpower(c)
    i = 0
    PM  = np.zeros((len(ST),))
    for k in ST:
        PM[i] = 10 * np.log10( sum([10**(0.1*P[k + j]) for j in range(-1,2,1)]))
        i += 1
    return PM

def get_hearing_threshold():
    Tq = np.load('Tq.npy', allow_pickle=True)
    #NaN values handling
    ind = 0
    while np.isnan(Tq[0, ind]):
        ind += 1
    Tq[0, 0:ind] = Tq[0, ind]
    while not(np.isnan(Tq[0, ind])):
        ind += 1
    Tq[0, ind:] = Tq[0, ind - 1]

    return Tq

def STreduction(ST, c, Tq):
    P = DCTpower(c)
    STr = []
    PMr = []
    for k in ST:
        if Tq[0, k] <= P[k]:
            STr.append(k)
            PMr.append(P[k])
    return STr, PMr

def SpreadFunc(ST, PM, Kmax):
    STlen = len(ST)
    MN = 1152 # Αν πάρουμε το Kmax ΝΑ ΤΟ ΑΛΛΑΞΟΥΜΕ
    SF = np.zeros((Kmax, STlen))
    # Sample Frequency
    fs = 44100
    # K to Frequencies
    f = [k*fs/(MN*2) for k in range(Kmax)]
    z = Hz2Barks(np.asarray(f))
    Dz = np.asarray([[z[i] - z[k] for k in ST] for i in range(Kmax)])
    # i every k corresponding to frequencies
    # j position of every masker in ST and PM arrays
    for i in range(Kmax):
        for j in range(STlen):
            if Dz[i, j] < -1 and Dz[i, j] >= -3:
                SF[i, j] = 17*Dz[i, j] - 0.4*PM[j] + 11
            elif Dz[i, j] < 0 and Dz[i, j] >= -1:
                SF[i, j] = (0.4*PM[j] + 6)*Dz[i, j]
            elif Dz[i, j] < 1 and Dz[i, j] >= 0:
                SF[i, j] = -17*Dz[i, j]
            elif Dz[i, j] < 8 and Dz[i, j] >= 1:
                SF[i, j] = (0.15*PM[j] - 17)*Dz[i, j] - 0.15*PM[j]
    return SF