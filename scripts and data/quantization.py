import numpy as np
import matplotlib.pyplot as plt

def critical_bands(K):
    MN = K
    cb = np.zeros(MN,)
    bands = create_crit_bands()
    bands_len = bands.shape[0]
    # Sample Frequency
    fs = 44100
    for i in range(MN - 1):
        # k to Frequencies
        f = i*fs/(MN*2)
        for j in range(bands_len):
            if f < bands[j, 3] and f >= bands[j, 2]:
                cb[i + 1] = bands[j, 0]
    return cb

def create_crit_bands():
    bands_len = 25
    bands = np.zeros((bands_len, 4))
    bands[:, 0] = np.arange(1, bands_len + 1, 1)
    bands[:, 1] = np.array([50, 150, 250, 350, 450, 570, 700, 840, 1000, 1175, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500, 19500]).reshape(bands_len, )
    bands[:, 2] = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]).reshape(bands_len, )
    bands[:, 3] = np.array([100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 22050]).reshape(bands_len, )
    return bands

def DCT_band_scale(c):
    # Cs and Cb first element is trash
    Kmax = len(c)
    # vector of bands for each k ceof
    cb = critical_bands(Kmax)
    bands_len = int(max(cb))  # = 25
    sc = []
    cs = np.zeros((Kmax,))
    for i in range(1, bands_len + 1):
        inds = np.where(cb == i)
        ci = c[inds]
        sci =max(abs(ci)**(3/4))
        sc.append(sci)
        cs[inds] = np.divide(np.multiply(np.sign(ci), abs(ci)**(3/4)), sci)
    return cs, sc

def quantizer(x, b):
    x_len = len(x)
    symb_index = np.zeros(x_len)
    zones_num = 2**b - 1
    wb = 2/(zones_num + 1)  # 2^(1-b)
    symbs = np.arange(-np.floor(zones_num / 2), np.ceil(zones_num / 2))
    for i in range(x_len):
        for s in range(zones_num):
            lower_bound = s*wb - 1
            upper_bound = lower_bound + wb
            if lower_bound == -wb:
                upper_bound = lower_bound + 2*wb
            elif lower_bound >= 0:
                lower_bound = upper_bound
                upper_bound = lower_bound + wb

            if x[i] >= lower_bound and x[i] <= upper_bound:
                symb_index[i] = symbs[s]
                break
    return symb_index

def dequantizer(symb_index, b):
    xh = np.zeros_like(symb_index)
    x_len = len(xh)
    symbs = symb_index + max(symb_index)
    zones_num = 2 ** b - 1
    wb = 2 / (zones_num + 1)  # 2^(1-b)
    for i in range(x_len):
        s = symbs[i]
        lower_bound = s*wb - 1
        upper_bound = lower_bound + wb
        if lower_bound == -wb:
            upper_bound = lower_bound + 2*wb
        elif lower_bound >= 0:
            lower_bound = upper_bound
            upper_bound = lower_bound + wb
        xh[i] = (lower_bound + upper_bound)/2
    return xh

def all_bands_quantizer(c, Tg):
    MN = len(Tg)
    cb = critical_bands(MN)
    max_bands = int(np.max(cb))
    symb_index = []
    B = np.zeros(max_bands)
    SF = np.zeros(max_bands)
    for i in range(1, max_bands + 1):
        b = 1
        c_band_inds = np.where(cb == i)[0]

        cs, sc = DCT_band_scale(c)
        cs_of_band = cs[c_band_inds]
        sc_of_band = sc[i - 1]

        while True:
            symb_index_c = quantizer(cs_of_band, b)
            c_h = dequantizer(symb_index_c, b)
            c_h_coeff = np.float64(np.sign(c_h) * np.cbrt(c_h * sc_of_band) ** 4)

            quant_error = abs(c[c_band_inds] - c_h_coeff)
            Pbi = 10 * np.log10(np.square(quant_error))
            Tgi = Tg[c_band_inds - 1]   #βάζω -1 επειδή το c_bands_inds ξεκινάει από το 1, ενώ το Tg από το 0
            # plt.plot(Pbi - Tgi)
            # string = "Error for band: " + str(i) + ", Bit Number: " + str(b)
            # plt.title(string)
            # plt.show()
            if all(Pbi <= Tgi):
                symb_index.append(symb_index_c)
                B[i-1] = b
                SF[i-1] = sc_of_band
                break
            else:
                b += 1
    return symb_index, SF, B

def all_bands_dequantizer(symb_index, B, SF):
    xh = []
    num_of_bands = len(B)
    for i in range(num_of_bands):
        b = B[i]
        symb_index_band = symb_index[i]
        xh_band = dequantizer(symb_index_band, b)
        xh_coeff = np.float64(np.sign(xh_band) * np.cbrt(xh_band * SF[i]) ** 4)
        xh.append(xh_coeff)
    xh = [item for sublist in xh for item in sublist]
    return xh