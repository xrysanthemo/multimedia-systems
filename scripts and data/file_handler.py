
def create_huff(filename:str):
    """
    Δημιουργεί το αρχείο στο οποίο θα αποθηκευτεί ο κώδικας Huffman
    """
    with open(filename, "w") as file:
        file.write("")

def write_huff(filename:str, frame_stream:str):
    """
    Γράφει τον κώδικα Huffman από κάθε frame σε αρχείο
    """
    with open(filename, "a") as file:
        file.write(frame_stream)

def read_huff(filename:str)->str:
    """
    Διαβάζει τα περιεχόμενα του αρχείου,
    που αποτελούν τον κώδικα Huffman όλης
    της διαδικασίας
    """
    with open(filename, "r") as file:
        contents = file.read()
    return contents