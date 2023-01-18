import numpy as np
from scipy.sparse import csr_matrix

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
    return ST




