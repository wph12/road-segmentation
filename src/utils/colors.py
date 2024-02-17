import math

from networkx.algorithms.coloring import greedy_color  # other graph coloring schemes are available too
import skimage as ski
import numpy as np
from skimage.color import color_dict

"""def get_colors(class_len):
    colors = [name for col, name in color_dict.items()]
    print("colors[0]:", colors[0])

    color_lst = []
    x = 0
    y = 0
    color_buffer = math.floor(len(colors) / class_len)
    print("class_len:", class_len)
    print("len(colors):", len(colors))
    print("color_buffer:", color_buffer)
    while y < class_len:
        print(f"idx {y}: {x}")
        color_lst.append(x)
        x += color_buffer
        y += 1
    # print("unique mask values:", np.unique(mask))
    _colors = [(colors[i][0] * 255, colors[i][1] * 255, colors[i][2] * 255) for i in color_lst]
    return _colors"""


def get_colors(binary=False):
    #print("getting color")
    # classes: {'background': 0, 'Wall': 1, 'Operable Wall': 2, 'Sliding Door': 3, 'Glass': 4, 'Window': 5, 'Door': 6}
    if binary:
        colors = [(0, 0, 0), (255, 255, 255)]
    else:
        colors = [
            (0, 0, 0), (38, 70, 83), (42, 157, 143), (233, 196, 106), (244, 162, 97), (231, 111, 81), (205, 180, 219),
            (255, 230, 167), (153, 88, 42), (67, 40, 24), (187, 148, 87), (111, 29, 27), (109, 104, 117), (181, 131, 141),
            (229, 152, 155), (255, 180, 162), (255, 205, 178), (255, 255, 255)
        ]
        colors = [(x[-1], x[1], x[0]) for x in colors]
    return colors
