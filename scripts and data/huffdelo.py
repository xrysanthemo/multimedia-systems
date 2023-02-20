import numpy as np
import heapq


def huff(run_symbols:np.ndarray)->(str, np.ndarray):
    """
    Κωδικοποίηση των RLE κατά Huffman με τη δημιουργία ενός Huffman Tree
    """
    run_symbols_str = run_symbols.astype(str)
    temp = np.char.add(run_symbols_str[:, 0], " ")
    run_symbols_str = np.char.add(temp, run_symbols_str[:, 1])


    symbols, counts = np.unique(run_symbols_str, return_counts=True)
    # Calculate the probability of each symbol
    probabilities = counts / len(run_symbols_str)

    run_symbols_unique = np.array(list(set(tuple(row) for row in run_symbols)))
    frame_symbol_prob = np.array([run_symbols_unique[:,0], run_symbols_unique[:,1], probabilities]).T

    heap = []
    for j in range(len(frame_symbol_prob)):
        symbol = str(int(frame_symbol_prob[j,0])) + " " + str(int(frame_symbol_prob[j,1]))
        weight = frame_symbol_prob[j, 2]
        heap.append([weight, [symbol, ""]])

    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Use the table to encode the symbols as a bitstream
    rle_huff = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    rle_huff_arr = np.asarray(rle_huff)
    frame_stream = []
    for rle_symbol in run_symbols_str:
        index = np.where(rle_huff_arr[:,0] == rle_symbol)[0][0]
        frame_stream.append(rle_huff_arr[index, 1])
    # from list of strings to one string
    frame_stream = "".join(frame_stream)
    return frame_stream, frame_symbol_prob

def ihuff(frame_stream:str, frame_symbol_prob:np.ndarray)->np.ndarray:
    """
    Αποκωδικοποίηση των RLE κατά Huffman με τη δημιουργία ενός Huffman Tree
    """
    # Build the huffman tree from the frame symbol probabilities
    heap = []
    for j in range(len(frame_symbol_prob)):
        symbol = str(int(frame_symbol_prob[j,0])) + " " + str(int(frame_symbol_prob[j,1]))
        weight = frame_symbol_prob[j,2]
        heap.append([weight, [symbol, ""]])
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    rle_huff = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    rle_huff_arr = np.asarray(rle_huff)

    # Decode the frame stream
    symbol = ""
    run_symbols_str = []
    for bit in frame_stream:
        symbol += bit
        if symbol in rle_huff_arr[:, 1]:
            run_symbols_str.append(rle_huff_arr[np.where(rle_huff_arr[:, 1] == symbol)[0][0]])
            symbol = ""

    # Delete huffman symbols column and convert to numpy array
    run_symbols = np.asarray(run_symbols_str)[:, :-1]

    #Split strings, and then refactor in type and dimensions same as those of initial run symbols
    run_symbols = np.array([np.array(item[0].split()) for item in run_symbols]).astype(int)
    return run_symbols


