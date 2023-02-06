import numpy as np

def RLE(symb_index,K):
    nonzero_ind = np.nonzero(symb_index)[0]
    nonzero_len = len(nonzero_ind)

    run_symbols = np.zeros((nonzero_len + 1, 2))
    index_offset = 1
    if not(symb_index[0] == 0):
        run_symbols = np.zeros((nonzero_len, 2))
        index_offset = 0

    for i in range(nonzero_len):
        zeros_num = 0
        zeros_start_ind = nonzero_ind[i] + 1
        while zeros_start_ind + zeros_num < K and symb_index[zeros_start_ind + zeros_num] == 0:
            zeros_num = zeros_num + 1
        run_symbols[i + index_offset, 0] = symb_index[nonzero_ind[i]]
        run_symbols[i + index_offset, 1] = zeros_num

    if index_offset == 1:
        zeros_in_start = K - (sum(run_symbols[:, 1]) + nonzero_len)
        run_symbols[0, 1] = zeros_in_start - 1

    return run_symbols.astype(int)

def iRLE(run_symbols,K):
    symb_index = np.zeros(K)
    len_run_sym = len(run_symbols[:, 0])
    ind = 0
    for i in range(len_run_sym):
        symb_index[i + ind] = run_symbols[i, 0]
        ind = run_symbols[i, 1] + ind
    return symb_index