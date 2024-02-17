import math
import os

import cv2
import numpy as np


def describe(gt_dir):
    gt_lst = [x for x in os.listdir(gt_dir) if ".jpg" in x or ".png" in x or ".npy" in x]
    # print("gt_lst:", gt_lst)
    percent_d = {}
    # classes = {'background': 0, 'Unlabelled': 1, 'Bedroom': 2, 'Foyer': 3, 'DiningRoom': 4, 'Wall': 5, 'Corridor': 6, 'Operable Wall': 7, 'Sliding Door': 8, 'Toilet': 9, 'Glass': 10, 'Window': 11, 'LivingRoom': 12, 'Kitchen': 13, 'Garage': 14, 'Stairs': 15, 'Door': 16}
    # classes = {'background': 0, 'Wall': 1, 'Operable Wall': 2, 'Sliding Door': 3, 'Glass': 4, 'one_room': 5,
               # 'Window': 6, 'Door': 7}
    classes = {"non_road": 0, "road": 1}
    res_d = {}
    # print(classes)
    for cls, idx in classes.items():
        percent_d[idx] = []
    # print(percent_d)
    for gt in gt_lst:
        pixel_count = 0
        lst = []
        if ".npy" in gt:
            gt = np.load(gt_dir + "/" + gt)
        else:
            gt = cv2.imread(gt_dir + "/" + gt)
        for cls, idx in classes.items():
            # print(len(gt) ** 3)
            if not np.unique(gt)[-1] == 0:
                c = np.count_nonzero(gt == idx)
                if idx == 0:
                    # print(c)
                    pass
                lst.append(c)
                pixel_count += c
        for idx, c in enumerate(lst):
            percent_d[idx].append(c / pixel_count)  # % of picture pixels in decimal
    # print("percent_d:", percent_d)
    for idx, percents in percent_d.items():
        res_d[idx] = sum(percents) / len(percents)
        print(f"{list(classes.keys())[idx]}: {res_d[idx] * 100}%")
        res_d[idx] = 1 / (math.ceil(res_d[idx]*10)/10)
    return res_d


"""# 128x128
background: 85.05125737281395%
Wall: 5.8341237157778405%
Operable Wall: 0.05272114607447506%
Sliding Door: 0.08177565041070332%
Glass: 0.0961350866027445%
Window: 0.2828474127631637%
Door: 1.2761782327290514%
Padding: 7.324961382828063%

# 256 x 256
background: 78.80887204299654%
Wall: 5.405924279575302%
Operable Wall: 0.04885157747907201%
Sliding Door: 0.0757736529389237%
Glass: 0.08907912810373694%
Window: 0.26208755143942475%
Door: 1.1825086129401003%
Padding: 14.126903154526909%

# 512x512
background: 41.72892540616745%
Wall: 2.8630422525141674%
Operable Wall: 0.04823956983000618%
Sliding Door: 0.032576930051090074%
Glass: 0.058968416747440765%
Window: 0.1375767078914201%
Door: 0.5459174926816228%
Padding: 54.5847532241168%
"""

if __name__ == "__main__":
    describe(r"/home/tester/jerome_ureca/uplan/temp/tiles/train/gt")
