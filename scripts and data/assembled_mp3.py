
from dct import iframeDCT, frameDCT
from psychoacoustics import DCTpower, Dksparse, STinit, MaskPower, get_hearing_threshold, STreduction, Hz2Barks, psycho
from quantization import critical_bands, DCT_band_scale, quantizer, dequantizer, all_bands_quantizer,  all_bands_dequantizer
from rle import RLE, iRLE
from huffdelo import huff, ihuff
from file_handler import write_huff, read_huff, create_huff
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from frame import frame_sub_analysis, frame_sub_synthesis
from scipy.io import wavfile
from nothing import donothing, idonothing
import math
import numpy as np

def MP3codec(wavin, h, M, N):
    # Διαβάζω το αρχείο .wav
    sr, data = wavfile.read(wavin)  # 514944
    # Κατασκευάζω τα φίλτρα ανάλυσης και σύνθεσης
    H = make_mp3_analysisfb(h, M)
    G = make_mp3_synthesisfb(h, M)
    # Δημιουργώ την παράμετρο για το l
    L = len(h)  # 512
    MN = M * N
    data_len = len(data)
    # Μέγεθος buffer
    xbuffer_size = (N - 1) * M + L
    ybuffer_rows = int((N - 1) + L / M)
    # Buffers
    xbuff = np.zeros([xbuffer_size])
    ybuff = np.zeros([ybuffer_rows, M])
    # Ορίζω το offset του buffer
    xoffset = xbuffer_size - MN
    yoffset = ybuffer_rows - N

    iters = math.ceil(data_len / MN)

    # Υπολογισμος συχνοτικών περιοχών
    D = Dksparse(MN)

    # Αρχικοποιώ xhat
    xhat = np.zeros(data.shape)

    # Create file for storing huffman encoding
    create_huff("huffman.txt")

    for i in range(iters):
        # Fill buffer
        xbuff[xoffset:xbuffer_size] = data[i * MN:(i + 1) * MN]
        # plt.plot(xbuff, 'g')
        # plt.plot(data[i * MN:(i + 1) * MN], 'r')
        # plt.title("Without Shift")
        # plt.show()
        # Frame Sub Analysis sto buffer mou
        Y = frame_sub_analysis(xbuff, H, N)
        # Shift xbuffer
        xbuff[0:xoffset] = xbuff[xbuffer_size - xoffset:]
        # plt.plot(xbuff, 'g')
        # plt.plot(data[i * MN:(i + 1) * MN], 'r')
        # plt.title("With Shift")
        # plt.show()
        # Επεξεργασία του frame
        # Yc = donothing(Y)

        # DCT
        c = frameDCT(Y)

        # Psycho
        # Υπολογισμός κατωφλίου ακουστότητας
        Tg = psycho(c, D)

        # Quantization
        symb_index_r, SF, B = all_bands_quantizer(c, Tg)
        print("frame: ", i, " bits: ", B)

        # RLE
        symb_index_r_flat = [int(item) for sublist in symb_index_r for item in sublist]
        run_symbols_rle = RLE(symb_index_r_flat, len(symb_index_r_flat))

        # Huffman
        frame_stream, frame_symbol_prob = huff(run_symbols_rle)
        write_huff("huffman.txt", frame_stream+"\n")

        # Αντιστροφή της διαδικασίας

        # iHuffman
        run_symbols_huff = ihuff(frame_stream, frame_symbol_prob)

        # iRLE
        symb_index_rle = iRLE(run_symbols_huff, len(symb_index_r_flat))  # TODO Calculate "len(symb_index_r_flat)" from run_symbols_huff

        # UNFLATTENING
        cb = critical_bands(MN)
        symb_index_unflat = []
        current_index = 0
        for j in range(1, int(max(cb) + 1)):
            count = np.count_nonzero(cb == j)
            symb_index_sublist = np.asarray(symb_index_rle[current_index:current_index + count])
            symb_index_unflat.append(symb_index_sublist)
            current_index = current_index + count
        # END OF UNFLATTENING

        # Dequantization
        ch = all_bands_dequantizer(symb_index_unflat, B, SF)
        ch = [0] + ch

        # iDCT
        Yh = iframeDCT(np.asarray(ch))

        ybuff[yoffset:ybuffer_rows, :] = Yh
        # Παραγωγή δειγμάτων synthesis
        Z = frame_sub_synthesis(ybuff, G)
        # Shift ybuffer
        ybuff[0:yoffset, :] = ybuff[ybuffer_rows - yoffset:, :]
        # Συσσώρευση σε xhat
        bound1 = i * N
        bound2 = (i + 1) * N
        xhat[(bound1 * M):(bound2 * M)] = Z

    # Write file to another file in our folder
    # ena teleutaio shift sto xhat
    val = xhat[0:xoffset]
    xhat[0:(len(xhat) - xoffset)] = xhat[xoffset:]
    xhat[(len(xhat) - xoffset):] = val
    wavfile.write("delofile_alla_kalutero.wav", sr, xhat.astype(np.int16))
    Y_tot = read_huff("huffman.txt")
    return xhat.astype(np.int16), Y_tot


def MP3cod(wavin, h,M,N):
    # Διαβάζω το αρχείο .wav
    sr, data = wavfile.read(wavin)  # 514944
    # Κατασκευάζω το φίλτρο ανάλυσης
    H = make_mp3_analysisfb(h, M)
    L = len(h)  # 512
    MN = M * N
    data_len = len(data)

    # Μέγεθος buffer
    xbuffer_size = (N - 1) * M + L
    # Buffers
    xbuff = np.zeros([xbuffer_size])
    # Ορίζω το offset του buffer
    xoffset = xbuffer_size - MN

    iters = math.ceil(data_len / (MN))

    # Αρχικοποιώ Y_tot, xhat
    Y_tot = np.zeros((N * iters, M))

    for i in range(iters):
        # Fill buffer
        xbuff[xoffset:xbuffer_size] = data[i * MN:(i + 1) * MN]
        # Frame Sub Analysis sto buffer mou
        Y = frame_sub_analysis(xbuff, H, N)
        # Shift xbuffer
        xbuff[0:xoffset] = xbuff[xbuffer_size - xoffset:]

        # Επεξεργασία του frame
        # DCT
        c = frameDCT(Y)

        # Psycho
        # Υπολογισμός κατωφλίου ακουστότητας
        Tg = psycho(c, D)

        # Quantization
        symb_index_r, SF, B = all_bands_quantizer(c, Tg)
        print("frame: ", i, " bits: ", B)

        # RLE
        symb_index_r_flat = [int(item) for sublist in symb_index_r for item in sublist]
        run_symbols_rle = RLE(symb_index_r_flat, len(symb_index_r_flat))

        # Huffman
        frame_stream, frame_symbol_prob = huff(run_symbols_rle)
        write_huff("huffman.txt", frame_stream+"\n")
    Y_tot = read_huff("huffman.txt")
    return Y_tot

def MP3decod(Y_tot, h, M, N):
    #TODO : /N AS DIVIDER IN TXT FILE, ITERS FROM TXT, APPEND IN XHAT INSEAD OF INITIALIZING
    sr = 44100
    data_len = Y_tot.shape[0] * Y_tot.shape[1]
    # Κατασκευάζω το φίλτρο σύνθεσης
    G = make_mp3_synthesisfb(h, M)
    L = len(h)  # 512
    MN = M * N

    # Μέγεθος buffer
    ybuffer_rows = int((N - 1) + L / M)
    # Buffers
    ybuff = np.zeros([ybuffer_rows, M])
    # Ορίζω το offset του buffer
    yoffset = ybuffer_rows - N

    iters = math.ceil(data_len / (MN))
    xhat = np.zeros(data_len)

    for i in range(0, iters):
        bound1 = i * N
        bound2 = (i + 1) * N
        Yc = Y_tot[bound1:bound2, :]
        # Αντιστροφή της διαδικασίας
        Yh = idonothing(Yc)
        ybuff[yoffset:ybuffer_rows, :] = Yh
        # Παραγωγή δειγμάτων synthesis
        Z = frame_sub_synthesis(ybuff, G)
        # Shift ybuffer
        ybuff[0:yoffset, :] = ybuff[ybuffer_rows - yoffset:, :]
        # Συσσώρευση σε xhat
        xhat[(bound1 * M):(bound2 * M)] = Z

    # Write file to another file in our folder
    wavfile.write("MYFILE_DECODER.wav", sr, xhat.astype(np.int16))
    return xhat.astype(np.int16)

