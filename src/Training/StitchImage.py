from src.utils.tiling_and_stitching import stitch


STITCH_IN_DIR = r"/Users/mouse/Documents/GitHub/uplan_project/dev/New Dataset/Data Set/tile/src"
#STITCH_IN_DIR = r"/Users/mouse/Documents/GitHub/uplan_project/dev/New Dataset/Data Set/tile/gt"
RESULTS_DIR = r"/Users/mouse/Documents/GitHub/uplan_project/dev/New Dataset/Data Set/visualize_all_gts/stitched/src"
#RESULTS_DIR = r"/Users/mouse/Documents/GitHub/uplan_project/dev/New Dataset/Data Set/visualize_all_gts/stitched/gt"
TILE_WIDTH = 512
TILE_HEIGHT = 512

if __name__ == "__main__":
    stitch(STITCH_IN_DIR, RESULTS_DIR, TILE_WIDTH, TILE_HEIGHT)