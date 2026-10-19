"""Material ids shared by the room, its props and the film, and the colour helper."""
import numpy as np

ENV, WALL, CEIL, FLOOR, WOOD, CUSHION, CAT, CUP, POT, LEAF, SHADE, DARK, BOOKS, PAINT = range(14)


def lin(*c):
    return np.array(c, np.float32) ** 2.2
PHONES, BOOK, TOMES = 14, 15, 16         # the listener's things: headphones, a paperback, books on the table (props.py)
