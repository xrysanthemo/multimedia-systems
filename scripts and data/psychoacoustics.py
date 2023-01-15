import numpy as np
from scipy.sparse import csc_array, csr_array, csr_matrix


def DCTpower(c):
    #ypologizei isxy se db apo suntelestes dct symfwna me auth th sxesh
    P = 10 * np.log10(np.square(abs(c)))
    return P

def Dksparse(Kmax): #Η είσοδος Kmax αντιστοιχεί στη μέγιστη διακριτη συχνότητα (εδώ Kmax = MN − 1)
    row = []
    col = []
    data =[]
    for k in range(0, Kmax): #k diakrites zwnes - grammes
        for j in range(0, Kmax): #sixnotita j e Dk - stiles
            if 2 < k and k < 282:
                row.append(k)
                col.append(2)
                data.append(Kmax/Kmax)
            elif 282 <= k and k < 570:
                for ind in range(2,14):
                    row.append(k)
                    col.append(ind)
                    data.append(Kmax/Kmax)
            elif 570 <= k and k < 1152:
                for ind in range(2,18):
                    row.append(k)
                    col.append(ind)
                    data.append(Kmax/Kmax)
    D = csr_matrix((data, (row, col)), shape=(Kmax, Kmax)) #.toarray()
    return D #έξοδος είναι sparse matrix Kmax * Kmax


d = Dksparse(11)
print(d)