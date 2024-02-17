import os

from tqdm import tqdm


def dir_builder(_dir):
    # print("Building", _dir)
    _dir_lst = _dir.split("/")
    d = "/" + _dir_lst[1]
    # print("root dir:", d)
    for i in tqdm(_dir_lst[2:]):
        # print("checking", d)
        if not os.path.isdir(d):
            os.mkdir(d)
        d += f"/{i}"
    if not os.path.isdir(_dir):
        os.mkdir(_dir)