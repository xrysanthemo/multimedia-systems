
def create_huff(filename):
    with open(filename, "w") as file:
        file.write("")

def write_huff(filename, frame_stream):
    with open(filename, "a") as file:
        file.write(frame_stream)

def read_huff(filename):
    with open(filename, "r") as file:
        contents = file.read()
    return contents