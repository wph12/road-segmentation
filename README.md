This project was completed with [@lemousehunter](https://github.com/lemousehunter) in fulfillment of EPFL's CS-433 course requirements. The report completed for this project can be found [here](https://github.com/wph12/road-segmentation/blob/13819bef8467e8e7acc9d10a78f85c3bafb67df0/CBAM2UNET_REPORT.pdf). 

# Introduction #
Road Segmentation is the process of identifying and delineating roads from given satellite imagery. In our project, we explored the addition of Convolutional Block Attention Modules (CBAM) to the U-Net for this task.

<img src="https://github.com/wph12/road-segmentation/assets/53130951/6866e3e6-1d30-4c0f-910c-ce9f88a21d69" height="250">
<img src="https://github.com/wph12/road-segmentation/assets/53130951/991cdd6b-3a94-4d8b-9980-27069b420844" height="250">



# How to Use #
## Training and generating output maps ##
Training and generating output maps works best on a linux OS, since Windows does not pair well with GPUs. Our model is quite resource intensive, and can take a significant amount of time to train. 
1. Clone the repository into a directory of your choice.
2. Set PYTHONPATH environment variable (on mac: type "export PYTHONPATH="path/to/parent/folder/of/src", on Windows, type "set PYTHONPATH="path/to/parent/folder/of/src")
3. In the root directory of this repo, run `pip install -r requirements.txt`. This assumes you are using python version 3.9 and above, and have pip installed.
4. Run train.py using `python src/Training/Train.py`. Alternatively, you can run `python Train.py` directly from the src/Training directory.
5. After the training has concluded, the model will generate the output feature maps, which can be found in the results/ directory.
6. You can then run `python generate_submission.py` in order to generate the submission file.

## Generating Predictions ##
To quickly generate the submission.csv results according to our model output, kindly follow these steps:
1. Clone the repository into a directory of your choice.
2. Set PYTHONPATH environment variable (on mac: type "export PYTHONPATH="path/to/parent/folder/of/src", on Windows, type "set PYTHONPATH="path/to/parent/folder/of/src") (if the steps for Training and generating output maps were not executed)
3. Either download your chosen model (either the [U-Net](https://github.com/wph12/road-segmentation/releases/download/models/unet.zip) or the [CBAM2U-Net](https://github.com/wph12/road-segmentation/releases/download/cbam2unet/cbamUnet.zip), or train a model (as detailed above in Training and generating output maps) 
4. In the root directory of this repo, run `pip install -r requirements.txt`. This assumes you are using python version 3.9 and above, and have pip installed.
5. Download the released model and extract it into a models directory. The path should look something like this: model/CbamUnet_last
6. `cd` into the src/inference/ directory.
7. Run `python run.py`. It should generate the needed submission csv file.


# References #
- O. Ronneberger, P.Fischer, and T. Brox, “U-net: Convolutionalnetworks for biomedical image segmentation,” in MedicalImage Computing and Computer-Assisted Intervention(MICCAI), ser. LNCS, vol. 9351. Springer, 2015, pp.234–241, (available on arXiv:1505.04597 [cs.CV]). [Online].Available: http://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a
- Y. Pang, Y. Li, J. Shen, and L. Shao, “Towards bridging semantic gap to improve semantic segmentation,” in 2019 IEEE/CVFInternational Conference on Computer Vision (ICCV), 2019,pp. 4229–4238.
- S. Wang, V. K. Singh, E. Cheah, X. Wang, Q. Li, S.-H. Chou, C. D. Lehman, V. Kumar, and A. E. Samir,“Stacked dilated convolutions and asymmetric architecturefor u-net-based medical image segmentation,” Computersin Biology and Medicine, vol. 148, p. 105891, 2022.[Online]. Available: https://www.sciencedirect.com/science/article/pii/S0010482522006308
- S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “Cbam: Convolutional block attention module,” 2018.
- Z. Zhang, Q. Liu, and Y. Wang, “Road extraction by deepresidual u-net,” IEEE Geoscience and Remote Sensing Letters,vol. 15, no. 5, p. 749–753, May 2018. [Online]. Available:http://dx.doi.org/10.1109/LGRS.2018.2802944

