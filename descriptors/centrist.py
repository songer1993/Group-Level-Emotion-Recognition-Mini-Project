import numpy as np

class CENTRIST:

    def __init__(self, divide = 4):
        self.divide = divide

    ''' The Census Transform

        Scan an 8 bit greyscale image with a 3x3 window
        At each scan position create an 8 bit number by comparing the value
        of the centre pixel in the 3x3 window with that of its 8 neighbours.
        The bit is set to 1 if the outer pixel >= the centre pixel

    '''

    def census_transform(self, gray):

        # Calulate size
        (w, h) = gray.shape

        # Convert image to Numpy array
        src_bytes = np.asarray(gray)

        # Initialize output array
        census = np.zeros((h - 2, w - 2), dtype='uint8')

        # centre pixels, which are offset by (1, 1)
        cp = src_bytes[1:h - 1, 1:w - 1]

        # offsets of non-central pixels
        offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

        # Do the pixel comparisons
        for u, v in offsets:
            census = (census << 1) | (src_bytes[v:v + h - 2, u:u + w - 2] >= cp)

        # Return census map
        return census



    # Receive directly the grayscale image
    def describe(self, gray):
        # Image size
        (w, h) = gray.shape
        block_size_w = w / self.divide
        block_size_h = h / self.divide

        hist = []

        # Crop out the window and calculate the histogram
        for r in range(0, w - block_size_w + 1, block_size_w):
            for c in range(0, h - block_size_h + 1, block_size_h):
                block = gray[r:r + block_size_w, c:c + block_size_h]
                census = self.census_transform(block)
                block_centrist, _ = np.histogram(census.flatten(), bins = 256)
                hist.append(block_centrist[1:-1].tolist())

        return [val for sublist in hist for val in sublist]
