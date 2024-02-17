import os
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

base_dir = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/ml-project-2-theasiandudes"
model_dir = base_dir + "/model"
history_fpath = model_dir + "/CbamUnet_history.pickle"
out_dir = base_dir + "/diagrams/history"


def plot_graph(y1, y2, title, xlabel, ylabel, y1_name, y2_name, fpath):
    plt.plot(y1)
    plt.plot(y2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend([y1_name, y2_name], loc="upper left")
    plt.savefig(fpath)


def visualize_graph(fpath, output_dir, training=True):
    with (open(fpath, "rb")) as openfile:
        history = pickle.load(openfile)
    if training:
        output_dir = output_dir + "/training"
    else:
        output_dir = output_dir + "/testing"
    output_dir += "/" + fpath.split("/")[-1].split(".")[0].replace("_history", "")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    plot_graph(history["loss"], history["val_loss"], "Loss v Epochs", "Epochs", "Loss", "TrainingLoss",
               "ValidationLoss", output_dir + "/loss.png")
    plt.clf()
    plot_graph(history["dice_coefficient"], history["val_dice_coefficient"], "DSC v Epochs", "Epochs", "DSC", "TrainingDSC",
               "ValidationDSC",
               output_dir + "/dsc.png")
    plt.clf()
    plot_graph(history["iou"], history["val_iou"], "IoU v Epochs", "Epochs", "IoU",
               "TrainingIoU",
               "ValidationIoU",
               output_dir + "/iou.png")

    print(history.keys())


if __name__ == "__main__":
    visualize_graph(history_fpath, out_dir)