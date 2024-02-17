This project was completed with [@lemousehunter](https://github.com/lemousehunter) in fulfillment of EPFL's CS-433 course requirements. The research paper completed for this project can be found [here](https://github.com/wph12/road-segmentation/blob/13819bef8467e8e7acc9d10a78f85c3bafb67df0/CBAM2UNET_REPORT.pdf). 

# Training and generating output maps #
Training and generating output maps works best on a linux OS, since Windows does not pair well with GPUs. Our model is quite resource intensive, and can take a significant amount of time to train. 
1. Clone the repository into a directory of your choice.
2. Set PYTHONPATH environment variable (on mac: type "export PYTHONPATH="path/to/parent/folder/of/src", on Windows, type "set PYTHONPATH="path/to/parent/folder/of/src")
3. In the root directory of this repo, run `pip install -r requirements.txt`. This assumes you are using python version 3.9 and above, and have pip installed.
4. Run train.py using `python src/Training/Train.py`. Alternatively, you can run `python Train.py` directly from the src/Training directory.
5. After the training has concluded, the model will generate the output feature maps, which can be found in the results/ directory.
6. You can then run `python generate_submission.py` in order to generate the submission file.

# Generating Predictions #
To quickly generate the submission.csv results according to our model output, kindly follow these steps:
1. Clone the repository into a directory of your choice.
2. Set PYTHONPATH environment variable (on mac: type "export PYTHONPATH="path/to/parent/folder/of/src", on Windows, type "set PYTHONPATH="path/to/parent/folder/of/src") (if the steps for Training and generating output maps were not executed)
3. Either download your chosen model (either https://github.com/CS-433/ml-project-2-theasiandudes/releases/download/model/unet.zip or https://github.com/CS-433/ml-project-2-theasiandudes/releases/download/model2/cbamUnet.zip), or train a model (as detailed above in Training and generating output maps) 
4. In the root directory of this repo, run `pip install -r requirements.txt`. This assumes you are using python version 3.9 and above, and have pip installed.
5. Download the released model and extract it into a models directory. The path should look something like this: model/CbamUnet_last
6. `cd` into the src/inference/ directory.
7. Run `python run.py`. It should generate the needed submission csv file.

