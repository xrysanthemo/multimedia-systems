import numpy as np

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
    Kmax = len(c)
    # vector of bands for each k ceof
    cb = critical_bands(Kmax)
    bands_len = int(max(cb))  # = 25
    sc = []
    cs = np.zeros((Kmax,))
    for i in sorted(list(set(cb[1:]))):
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
    bands = create_crit_bands()
    MN = len(Tg)
    cb = critical_bands(MN)
    max_bands = int(np.max(cb) + 1)
    symb_index = []
    B = []
    SF =[]
    for i in range(1,max_bands):
        Pb = np.ones(len(c)) * np.inf
        while not(np.all(Pb[np.where(cb == i)] < Tg[np.where(cb == i)[0] - 1])):
            b = 1
            c_of_band = c[np.where(cb == i)[0] - 1]
            cs, sc = DCT_band_scale(c_of_band)
            symb_index_c = quantizer(cs, b)
            c_h = dequantizer(symb_index_c, b)
            c_h_coeff = np.sign(c_h) * ((c_h * sc[0])**(4/3))
            quant_error = np.abs([np.where(cb == i)[0] - 1] - c_h_coeff)
            Pb = 10 * np.log10(quant_error**2)
            b += 1
        symb_index.append(symb_index_c)
        B.append(b)
        SF.append(sc[0])
    return symb_index, SF, B