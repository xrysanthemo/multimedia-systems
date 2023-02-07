import numpy as np
import heapq

def invert_tree(node, prefix, code_table):
    if isinstance(node[1], str):
        # This is a leaf node, so add the symbol and its Huffman code to the table
        code_table[node[1]] = prefix
    else:
        # This is an internal node, so traverse its children
        invert_tree(node[1][0], prefix + [0], code_table)
        invert_tree(node[1][1], prefix + [1], code_table)

def huff(run_symbols):
    run_symbols_str = run_symbols.astype(str)
    run_symbols_str = np.char.add(run_symbols_str[:, 0], run_symbols_str[:, 1])

    symbols, counts = np.unique(run_symbols_str, return_counts=True)
    # Calculate the probability of each symbol
    probabilities = counts / len(run_symbols_str)
    frame_symbol_prob = np.array([symbols, probabilities]).T

    # Huffman encode
    # Create a list of tuples, where each tuple consists of a symbol and its probability
    nodes = list(zip(symbols, probabilities))

    # Sort the list of tuples in ascending order of probabilities
    heapq.heapify(nodes)

    while len(nodes) > 1:
        # Take the two nodes with the lowest weights
        node1 = heapq.heappop(nodes)
        node2 = heapq.heappop(nodes)

        # Create a new node with these two nodes as its children
        new_node = (node1[1] + node2[1], [node1, node2])

        # Add the new node back to the list, sorted by its weight
        heapq.heappush(nodes, new_node)

    # The remaining node is the root of the Huffman tree
    root = nodes[0]

    # Traverse the tree to create a table that maps each symbol to its Huffman code
    code_table = {}
    invert_tree(root[1], [], code_table)

    # Use the table to encode the symbols as a bitstream
    frame_stream = [code_table[symbol] for symbol in symbols]

    return frame_stream, frame_symbol_prob


# def ihuff (frame_stream, frame_symbol_prob):
#     def huffman_decode(encoded_symbols, code_table):
#         # Create a table that maps each code to its corresponding symbol
#         inverted_table = {}
#         for symbol, code in code_table.items():
#             inverted_table[tuple(code)] = symbol
#
#         # Decode the symbols
#         decoded_symbols = []
#         for encoded_symbol in encoded_symbols:
#             decoded_symbols.append(inverted_table[tuple(encoded_symbol)])
#
#         return decoded_symbols
#
#     # Example usage
#     encoded_symbols = huffman_encode(symbols, probabilities)
#     code_table = huffman_encode(symbols, probabilities)
#     decoded_symbols = huffman_decode(encoded_symbols, code_table)
#     print(decoded_symbols)


