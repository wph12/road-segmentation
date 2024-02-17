import glob
import math
import os

import cv2
import numpy as np
from tqdm import tqdm


def pad_img(tile, o_shape, n_shape):
    if len(tile.shape) == 2:
        n_shape = n_shape[:-1]
    #padded = np.ones(n_shape) * num_classes
    padded = np.zeros(n_shape)

    # compute center offset
    y_center = (n_shape[0] - o_shape[0]) // 2
    x_center = (n_shape[1] - o_shape[1]) // 2
    # print("o_shape:", o_shape)
    # print("n_shape:", n_shape)
    # copy img image into center of result image
    x_start = x_center
    x_end = x_center + o_shape[1]

    y_start = y_center
    y_end = y_center + o_shape[0]

    padded[y_start:y_end, x_start:x_end] = tile

    return padded, (y_start, y_end, x_start, x_end)


def tile_img(img, tile_shape, makeLastPartFull=True) -> [[np.ndarray, ..., np.ndarray], ..., [np.ndarray, ..., np.ndarray]]:
    tileSizeX, tileSizeY, dim = tile_shape
    numTilesX = math.ceil(img.shape[1] / tileSizeX)
    # print(f"numTilesX: {numTilesX}")
    numTilesY = math.ceil(img.shape[0] / tileSizeY)
    # print(f"numTilesY: {numTilesY}")

      # in case you need even size

    tiles = []

    for col_num in range(numTilesX):
        col = []
        for row_num in range(numTilesY):
            coords = None

            startX = col_num * tileSizeX
            endX = startX + tileSizeX

            startY = row_num * tileSizeY
            endY = startY + tileSizeY

            if endY > img.shape[0]:
                endY = img.shape[0]

            if endX > img.shape[1]:
                endX = img.shape[1]
            # print(f"col_num: {col_num}, startX: {startX}, endX: {endX}, delta: {endX - startX}")
            # print(f"row_num: {row_num}, startY: {startY}, endY: {endY}, delta: {endY - startY}")
            currentTile = img[startY:endY, startX:endX]
            if makeLastPartFull is True and (col_num == numTilesX - 1 or row_num == numTilesY - 1):
                currentTile, coords = pad_img(currentTile, currentTile.shape, tile_shape)
            col.append((currentTile, coords))
        tiles.append(col)
    return tiles


def _stitch(_dir, results_dir, tilenames, tile_width, tile_height, class_len):
    tilenames.sort()
    _tilenames = []
    max_r = max_c = 0
    tiles_d = {}  # {row_num: {col_num: tile}}
    for tilename in tilenames:
        if "merged" not in tilename:
            tilename, ext = tilename.split(".")
            _lst = tilename.split("_coords")
            if len(_lst) == 1:
                fpath, coords = _lst[0], None
            else:
                fpath, coords = _lst

            fpath = fpath.split("/")[-1]
            # print("fpath:", fpath)
            fname = "_".join(fpath.split("_")[:2])
            # print("fname:", fname)
            r = int(fpath.split("_")[-2].replace("r", ""))
            c = int(fpath.split("_")[-1].replace("c", ""))

            if r > max_r:
                max_r = r
            if c > max_c:
                max_c = c

            if "jpg" in ext or "png" in ext:
                tile = cv2.imread(tilename + f".{ext}")
            else:
                tile = np.load(tilename + f".{ext}")
            # print("ingested tile shape:", tile.shape)
            if len(tile.shape) == 2:
                # print("stacking...")
                tile = np.dstack((tile, tile, tile))
            if coords is not None:
                c_lst = coords.split("_")[1:]
                # print("c_lst:", c_lst)
                c0 = int(c_lst[0].replace("x0", ""))
                c1 = int(c_lst[1].replace("x1", ""))
                d0 = int(c_lst[2].replace("y0", ""))
                d1 = int(c_lst[3].replace("y1", ""))
                coords = (c0, c1, d0, d1)

                # print(f"cropping: tile[{d0}:{d1}, {c0}:{c1}]")
                tile = tile[d0:d1, c0:c1]

            # print("tile.shape:", tile.shape)

            """if len(tile.shape) > 3:
                if tile.shape[0] == 0:
                    print("filling empty array with zeroes")
                    tile = np.zeros((1, tile.shape[1], tile.shape[2], tile.shape[3]))
                tile = tile[0, :, :, :]
                print("new tile.shape:", tile.shape)
            if tile.shape[-1] > 3:
                tile = np.argmax(tile, axis=2)
                print("new new tile.shape:", tile.shape)
                tile = np.dstack((tile, tile, tile))
                print("new new new tile.shape:", tile.shape)"""
            # print("tile.shape:", tile.shape)
            y, x, dim = tile.shape
            tile = {"tile": tile, "x": x, "y": y}
            if r not in tiles_d.keys():
                tiles_d[r] = {}

            if c not in tiles_d[r].keys():
                tiles_d[r][c] = tile

            # print(f"r: {r}, c: {c}")
            # print("fpath:", fpath)
            # print("fpath, coords: ", fpath, coords)
            # max_x: 1597 max_y: 799
            # src shape: (799, 1597, 3)

    # print("tiles:", tiles_d)

    # print("max_r:", max_r)
    # print("max_c:", max_c)

    max_x = max_c * tile_width + tiles_d[0][max_c]["x"]
    max_y = max_r * tile_width + tiles_d[max_r][0]["y"]

    # print("x:", tiles_d[0][max_c]["x"])
    # print("y:", tiles_d[max_r][0]["y"])

    # print("max_x:", max_x, "max_y:", max_y)
    # print(tilenames)

    img = np.zeros((max_y, max_x, 3))

    for r, col in tiles_d.items():
        for c, tile in col.items():
            _tile = tile["tile"]
            x = tile["x"]
            y = tile["y"]
            x0 = c * tile_width
            x1 = x0 + x
            y0 = r * tile_height
            y1 = y0 + y
            # print(f"row: {r}, col: {c}, x0: {x0}, x1: {x1}, y0: {y0}, y1: {y1}, delta: ({y}, {x}, 3)")
            # print(_tile.shape)
            img[
            y0:y1,
            x0:x1,
            :] = _tile
    # print(img.shape)
    if class_len:
        img = (img / class_len * 255).astype("uint8")[:, :, 0]
    out_fname = results_dir + f"/{'_'.join(tilenames[0].split('/')[-1].split('_')[:1]) + '_merged.png'}"
    # print("visualized", out_fname)
    cv2.imwrite(out_fname, img)


def stitch(_dir, results_dir, tile_width, tile_height, class_len=None):
    print("stitching...")
    print("DirExists:", os.path.isdir(_dir))
    tilenames = [x for x in list(os.listdir(_dir)) if ".jpg" in x or ".png" in x]

    filenames = {}
    # print("tilenames:", tilenames)
    for tile in tqdm(tilenames):
        # print("tile:", tile)
        if tile.split("_")[0] not in filenames.keys():
            #print("t:", tile)
            #print("tilename:", tile.split("/")[-1].split("_")[0])
            pathname = os.path.join(_dir, tile.split("/")[-1].split("_")[0] + "*")
            #print("pathname:", pathname)
            filenames[tile.split("/")[-1].split("_")[0]] = [x for x in list(glob.glob(pathname)) if "merged" not in x]
    print("len filenames:", len(filenames.keys()))
    print("filenames:", filenames.keys())
    for _, fileset in filenames.items():
        _stitch(_dir, results_dir, fileset, tile_width, tile_height, class_len)
