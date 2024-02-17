from keras.utils import plot_model

from src.Architectures.Models.CBAMUnet import CbamUnet

model = CbamUnet(64, 2, 0.3, 400, 3)()
plot_model(model, to_file='/Users/mouse/Documents/GitHub/CS-433_ML_Projects/ml-project-2-theasiandudes/diagrams/architectures/CbamUnet.jpg')

