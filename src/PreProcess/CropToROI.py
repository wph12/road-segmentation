import networkx as nx
import numpy as np
import cv2
import os

from tqdm import tqdm

#IMG_IN_DIR = "/dev/New Dataset/Data Set/Processed/test/src"
#GT_IN_DIR = "/dev/New Dataset/Data Set/Processed/test/gt"
#IMG_OUT_DIR = "/dev/New Dataset/Data Set/Processed/new/test/src"
#GT_OUT_DIR = "/dev/New Dataset/Data Set/Processed/new/test/gt"

IMG_IN_DIR = "/Users/mouse/Documents/GitHub/uplan_project/dev/New Dataset/Data Set/ProcessedOneRoom/src"
GT_IN_DIR = "/Users/mouse/Documents/GitHub/uplan_project/dev/New Dataset/Data Set/ProcessedOneRoom/gt"
IMG_OUT_DIR = "/Users/mouse/Documents/GitHub/uplan_project/dev/New Dataset/Data Set/Cropped/OneRoom/src"
GT_OUT_DIR = "/Users/mouse/Documents/GitHub/uplan_project/dev/New Dataset/Data Set/Cropped/OneRoom/gt"


#FNAME = "uplan-21.png"


def remove_contained_boxes(bounding_boxes):
    result_boxes = []

    for i, box1 in enumerate(bounding_boxes):
        is_contained = False

        for j, box2 in enumerate(bounding_boxes):
            if i != j:
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2

                # Check if box1 is completely contained within box2
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    is_contained = True
                    # print("is contained")
                    break

        if not is_contained:
            result_boxes.append(box1)

    return result_boxes


def find_union_of_bounding_boxes(bounding_boxes):
    if not bounding_boxes:
        return None

    # Initialize the union box with the first bounding box
    x_union, y_union, w_union, h_union = bounding_boxes[0]

    for box in bounding_boxes[1:]:
        x, y, w, h = box

        # Calculate the coordinates and dimensions of the union box
        x_union = min(x_union, x)
        y_union = min(y_union, y)
        w_union = max(x_union + w_union, x + w) - x_union
        h_union = max(y_union + h_union, y + h) - y_union

    return (x_union, y_union, w_union, h_union)


def group_adjacent_bounding_boxes(bounding_boxes, threshold=0):
    def are_adjacent(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the horizontal and vertical distances between the centers of the bounding boxes
        dx = abs((x1 + w1 / 2) - (x2 + w2 / 2))
        dy = abs((y1 + h1 / 2) - (y2 + h2 / 2))

        # Calculate the minimum distance between the centers of the bounding boxes
        min_dist = min(w1, w2) / 2 + min(h1, h2) / 2

        return dx < min_dist + threshold and dy < min_dist + threshold

    # Create a graph to represent adjacency relationships
    G = nx.Graph()

    # Add nodes for each bounding box
    for i, box1 in enumerate(bounding_boxes):
        G.add_node(i)

    # Add edges to the graph for adjacent bounding boxes
    for i, box1 in enumerate(bounding_boxes):
        for j, box2 in enumerate(bounding_boxes):
            if i < j and are_adjacent(box1, box2):
                G.add_edge(i, j)

    # Find connected components in the graph
    groups = list(nx.connected_components(G))

    # Extract grouped bounding boxes
    grouped_boxes = []
    for group in groups:
        grouped_boxes.append([bounding_boxes[i] for i in group])

    return grouped_boxes


def crop_image(image, bounding_box):
    x, y, width, height = bounding_box
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image


def remove_small_bounding_boxes(bounding_boxes, min_size):
    filtered_boxes = []

    for box in bounding_boxes:
        x, y, width, height = box
        box_size = width * height

        if box_size >= min_size:
            filtered_boxes.append(box)

    return filtered_boxes


def binarize_mask(mask):
    # Create a binary mask where values between 0 and 7 are set to 0 (black),
    # and other values are set to 255 (white)
    binary_mask = np.where(mask != 0, 1, 0).astype(np.uint8)

    return binary_mask


def crop(fname, img_in_dir, gt_in_dir, img_out_dir, gt_out_dir, buffer_amt):
    # print("cropping")
    img_in_fpath = img_in_dir + "/" + fname + ".jpg"

    if not os.path.isfile(img_in_fpath):
        img_in_fpath = img_in_fpath.replace(".jpg", ".png")

    image = cv2.imread(img_in_fpath)

    gt_in_fpath = gt_in_dir + "/" + fname + ".npy"
    if not os.path.isfile(gt_in_fpath):
        gt_in_fpath = gt_in_dir + "/" + fname + ".png"
        gt = cv2.imread(gt_in_fpath)
    else:
        gt = np.load(gt_in_fpath)

    #print("gt shape:", gt.shape)
    #print("gt unique:", np.unique(gt))

    # binarising
    _gt = gt[:, :, 0]
    #gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    _gt = binarize_mask(_gt)
    # print("np.unique:", np.unique(_gt))

    # applying morphological operations to dilate the image
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(_gt, kernel, iterations=3)

    # finding contours, can use connectedcomponents aswell
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours:", contours)

    # converting to bounding boxes from polygon
    contours = [cv2.boundingRect(cnt) for cnt in contours]

    #contours = remove_small_bounding_boxes(contours, 50)
    contours = remove_contained_boxes(contours)
    contours = group_adjacent_bounding_boxes(contours)
    # drawing rectangle for each contour for visualising
    img_counter = 0
    for c in contours:
        #print("contours:", c)
        if len(c) == 1:
            c = c[0]
        else:
            if len(c) > 2:
                # print("more than 2")
                pass
            c = find_union_of_bounding_boxes(c)
        x, y, w, h = c

        c = (x, y, w, h)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _image = crop_image(image, c)

        #print(f"{fname} unique:", np.unique(image))
        img_out_fpath = img_out_dir + f"/{fname.split('.')[0]}-{img_counter}.png"
        cv2.imwrite(img_out_fpath, _image)

        _gt = crop_image(gt, c)
        gt_out_fpath = gt_out_dir + f"/{fname.split('.')[0]}-{img_counter}.npy"
        np.save(gt_out_fpath, _gt)
        # print("gt out:", gt_out_fpath)
        img_counter += 1


if __name__ == "__main__":
    print("len gt dir:", len(os.listdir(GT_IN_DIR)))
    for FNAME in tqdm(os.listdir(GT_IN_DIR)):
        if ".npy" in FNAME:
            crop(FNAME.split(".")[0], IMG_IN_DIR, GT_IN_DIR, IMG_OUT_DIR, GT_OUT_DIR, 5)