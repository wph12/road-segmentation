import pickle

UNET = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/ml-project-2-theasiandudes/model/Unet_history.pickle"
CBAMUNET = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/ml-project-2-theasiandudes/model/CbamUnet_history.pickle"


def compute_metrics(metrics_file):
    with (open(metrics_file, "rb")) as openfile:
        history = pickle.load(openfile)
        print(history.keys())
        iou = history['val_iou']
        dsc = history['val_dice_coefficient']

        _i = []
        avg_iou = sum(iou) / len(iou)
        for i in iou:
            _i.append(abs(i - avg_iou))
        avg_iou_dev = sum(_i) / len(_i)
        _d = []
        avg_dsc = sum(dsc) / len(dsc)
        for d in dsc:
            _d.append(abs(d - avg_dsc))
        avg_dsc_dev = sum(_d) / len(_d)

        if "Cbam" in metrics_file:
            print("CbamUnet____________")
        else:
            print(print("Unet____________"))

        print(f"avg_iou: {avg_iou} +- {avg_iou_dev}")
        print(f"avg_dsc: {avg_dsc} +- {avg_dsc_dev}")


if __name__ == "__main__":
    compute_metrics(UNET)
    compute_metrics(CBAMUNET)