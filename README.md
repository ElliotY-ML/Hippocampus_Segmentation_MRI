# Quantifying Alzheimer's Disease Progression Through Automated Measurement of Hippocampal Volume
This repository contains a completed cap-stone project for Udacity's "Applying AI to 3D Medical Imaging Data" course, 
part of the AI for Healthcare Nanodegree program.  It has been reviewed by Udacity instructors and met project specifications.

## Introduction  

Alzhiemer's Disease (AD) is a degenerative brain disease that affects an estimated 5.8 million Americans age 65 and older in 2020.  
It is thought that AD begins 20 years or more before symptoms arise, with progressive brain changes that are unnoticeable to the affected person.  As the disease progresses, nerve cells (neurons) in parts of the brain involved with thinking, learning, and memory functions are damaged and destroyed.  
After years of brain changes, individuals experience symptoms such as memory loss, loss of language function, and other manifestations.  AD is the most common cause of dementia [1].

The Alzheimer's Association (AA) "2020 Alzheimer's Disease Facts and Figures" estimates that the number of Americans with Alizheimers may triple by 2050 [1].  
With such a staggering future care need, projections show that there will be a shortage of front-line primary care physicians (PCP), neurologists, and other specialists who provide critical expertise in dementia diagnosis and care [2].

Currently, an MRI exam is one of the most advanced methods to quantify Alzheimer's.  Studies have shown that meausrement of hippocampal volume from MRI exams is useful to diagnose and track progression of several brain diseases, including AD.  AD patients have shown a reduced hippocampus volume.
Quantifying disease progression over time can help direct therapy and disease management. However, the process to measure the hippocampus using MRI scans is very time consuming.  Each 3D MRI scan volume contains several dozen 2D images slices.  At each 2D image slice, the hippocampus must be correctly identified and traced.

AI software can provide a practical solution to quantify hippocampal volume from MRI scans.  Deep learning algorithms for computer vision segmentation tasks introduce new avenues to automate the identification of objects and trace objects in an image.   
For this project, a deep learning segmentation model was created to identify hippocampus structures in brain MRI scans on volume pixel (voxel) level.  The identified hippocampus voxels are translated to a physical measurement, mm^3.

The intention of this software is to be integrated into a Picture Archiving and Commonication System (PACS) whereby this software will automatically calculate hippocampal volumes of new MRI studies as the studies are committed in the clinical imaging archive.
This software will eliminate the tedious hippocampus measurement task from physicians and quickly provide physicians with an accurate measurement.  The software will also provide a consistent method to trace the hippocampus structure, whereas there may be variability between clinicians in the tracement task. 

This project is broken into three sections and are located in separate folders:
- Section 1 Curating a Dataset of Brain MRIs: Analyze Medical Decathlon dataset metadata, analyze & visualize image volumes & corresponding labels, and identify & remove data that is not of a brain MRI.  
- Section 2 Training a segmentation CNN model: Image volume extraction from NIFTI files, image volume pre-processing, split dataset using Scikit-Learn, build & train a UNet Fully Convoluted Neural Network (FCN) with PyTorch, 
and evaluate model performance metrics - overall Dice Similarity Coefficent & Jaccard Index.  
- Section 3 Integrating into a Clinical Network:  Simulate DICOM Message Service Element (DIMSE), where a computer containing a copy of the Sections 2 segmentation algorithm listens for PACS file transfer, and requests a copy of the transferred file to execute inference and provide hippocampus measurement.

**References**
[1] Alzheimer’s Association. "2020 Alzheimer’s Disease Facts and Figures", Alzheimers & Dementia, 2020;16(3):391+. [LINK](https://www.alz.org/media/Documents/alzheimers-facts-and-figures_1.pdf)
[2] "Primary Care Physicians on the Front Lines of Diagnosing and Providing Alzheimer’s and Dementia Care: Half Say Medical Profession Not Prepared to Meet Expected Increase in Demands". www.alz.org, 2020 [LINK](https://www.alz.org/news/2020/primary-care-physicians-on-the-front-lines-of-diag)
# for MRI to introduce measurement of Hippocampus for longitudinal care for alzheimer's patients.

**Part 1: Curating a Dataset for Machine Learning Training and Validation**  

The human brain has two hippocampi, one in the left hemisphere and one in the right hemisphere of the brain.  Udacity provided this project's dataset that consists of cropped regions around the right hippocampus.
The dataset contains MRI scan volumes that may be for brain studies and other types of studies.  This Section of the project reviews the given dataset to clean the dataset, and retreive only Brain MRI scan volumes.

Inputs: 
- `/data/TrainingSet/images` contains 262 NIFTI files for MRI Scan Volumes
- `/data/TrainingSet/labels` contains 262 NIFTI files for corresponding Segmentation label masks

Outputs: 
- `/Section 1 EDA/out/images` contains 260 NIFTI files that are Brain MRI Scan Volumes
- `/Section 1 EDA/out/labels` contains 260 NIFTI files that are Brain Hippocampus Segmentation label masks

Instructions:
1. This section of the project was completed in the Jupyter Notebook `/Section 1 EDA/Final Project EDA.ipynb`.  Open this notebook to start.
2. The first step is to create lists for images and labels filepaths.
3. Using the NiBabel python library, the NIFTI files are extracted.
4. For a handful of files, visualize select 2D slices from each 3D MRI volume.
5. Explore the metadata from NIFTI file headers.  This contains information about MRI volume dimensions, MRI scanner settings, and voxel dimensions.
6. Use metadata, image data, and segmentation mask data to find MRI volumes that do not appear similar to most of the dataset.
7. Use voxel information and segmentation mask to calculate Hippocampus volume per MRI scan.  Investigate MRI scans that are not in a typical range of Hippocampus sizes.
8. After identifying non-Brain MRI files, use `shutil` to copy the NIFTI image and label volumes into the `/Section 1 EDA/out` folder.


**Part 2: Train U-Net Fully Convoluted Network for Brain Segmentation**

In Section 2, PyTorch is used for training a model with the U-Net convolutional neural network architecture from the University of Freiburg [1] for segmentation of Brain MRIs and identify the right hippocampus.  
Cleaned data from Section 1 is the input into Section 2.   The directory `/Section 2 Train_Eval_Model/src` contains the source code that forms the machine learning pipeline.  

Inputs:
- '/Section 2 Train_Eval_Model/images` contains 260 NIFTI Files containing cropped Brain MRI volumes 
- '/Section 2 Train_Eval_Model/labels` contains 260 NIFTI Files containing Right Hippocampus Labels 
 
Outputs:  
*Stored in '/Section 2 Train_Eval_Model/out' in folders named "YYYY-MM-DD_Basic-unet":
- Trained model and weights for segmentation of Hippocampus in brain MRI volumes stored in file named `model.pth`.
- Model performance metrics information, DICE Similarity Coefficient and Jaccard Index, stored in `results.json` file.

Instructions:  
1. Run script 'run_ml_pipeline.py' in `/Section 2 Train_Eval_Model/src`.  It will call and execute methods from modules contained in the `/src/` tree to extract & pre-process NIFTI Brain MRI volumes, complete model training, and evaluate performance.
2. `run_ml_pipeline.py` has hooks to log progress to Tensorboard.  To see the Tensorboard output, launch Tensorboard executable from the same directory where `run_ml_pipeline.py` is location by using the command:
> tensorboard --logdir runs --bind_all
3.  Tensorboard will write logs into the director called `runs`.  View the progress by opening a browser and navigate to port 6006 of the machine where you are running it.

In a completed model run, the model achieved performance of **Overall Mean Dice Similarity Coefficent 0.906** and **Overall mean Jaccard Index 0.83**.  This meets requirements for Dice Coefficient >0.90 and Jaccard Index >0.80.

**References**
[1]  Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015, available at arXiv:1505.04597 [cs.CV] 


**Part 3: Simulate Integration of Segmentation CNN into DIMSE**

In Section 3, the segmentation CNN from Section 2 will be integrated into a simulated clinical network.  This AI product will automatically compute hippocampus volume for brain MRI scans, and provide this information to clinicians in a DICOM report.  

DIMSE Simulation Setup:  
![Clinical Network Setup](/data/readme.img/network_setup.png)

Network Object:										Script to Simulate Network Object:
1.  Picture Archiving & Communications System (PACS) server 	1. Orthanc DICOM server [1]
2.  MRI Scanner 												2. `section3/src/deploy_scripts/send_volume.sh`.  It will initiate a file transfer to the Orthanc.
3.  Viewer System 												3. OHIF Viewer [2].  It connects to the Orthanc server using DicomWeb and is serving a web application on port 3000. 
4.  AI Server containing Segmentation software					4. `section3/src/deploy_scripts/start_listener.sh`.  It will copy everything it receives into a folder specified in the script.

1.  The PACS server is central to clinical settings.  It receives & archives all medical images and allows connected computers to request & send image files.  The Orthanc software, by Sébastien Jodogne, is a standalone DICOM server which allows the similation of a PACS server [1].
 For this project, the Orthanc is listening to DICOM DIMSE requests on port 4242 and has a DicomWeb interface that is open at port 8042.  It is also running a model that sends everything it receives to an AI server.
2.  The MRI Scanner will send entire studies to the Picture Archiving and Communication System (PACS) Orthanc server after completing a scan.  The script will simulate the archive transfer.
3.  The Viewer system represents workstations that clinicians use to retreive and view studies from PACS.  The OHIF is viewer is software for viewing medical studies.  It is connecting to the Orthanc server using DicomWeb and is serving a web application on port 3000.
4.  An AI server is responsible for listening to PACS ports for incoming MRI studies.  When it detects that an MRI study is sent, the AI server will request a copy from the PACS server.  Once the MRI study is received on the AI server, the brain MRI scan will be processed by segmentation software and the hippocampus volume will be calculated from the determined hippocampus mask.  

Inputs:  
- A file transfer of a Brain MRI scan.

Outputs:
- A DICOM Report displaying Total Hippocampal Volume, Anterior Hippocampal Volume, Posterior Hippocampal Volume, and Axial views (head to top view) at three depths.


Instructions:

1.  Set up Orthanc.  Open a terminal and enter the following:
`bash launch_orthanc.sh` or `./launch_orthanc.sh`. Don't close this terminal.  
Wait for it to complete, with the last line being something like
`W0509 05:38:21.152402 main.cpp:719] Orthanc has started` and/or you can verify that Orthanc is working by running `echoscu 127.0.0.1 4242 -v` in a new terminal.
2.  Set up OHIF.  Open a new terminal and enter the following
`bash launch_OHIF.sh` or `./launch_OHIF.sh`. Don't close this terminal
Wait for it to complete, with the last line being something like
`@ohif/viewer: ℹ ｢wdm｣: Compiled with warnings.`  
You will then want to enter the Desktop with the bottom right hand corner.
-  OHIF should automatically open in a Web Browser but if not you can paste `localhost:3005` into the address bar of a Web browser window.
-  orthanc isn't necessary to open but if you need it you can access it can paste `localhost:8042` into the address bar of a Web browser window.
2. Open a terminal and cd to `Section 3 Simulate DIMSE/src`.  Run `start_listener.sh`.  Keep this terminal open.
3. Open another terminal and Run `Section 3 Simulate DIMSE/src/deploy_scripts/send_volume.sh` to simulate sending of MRI studies to the PACS. 
4. Run `Section 3 Simulate DIMSE/src/inference_dcm.py` calls and executes methods to 

**References**
[1] Jodogne, S. The Orthanc Ecosystem for Medical Imaging. J Digit Imaging 31, 341–352 (2018). [Link](https://doi.org/10.1007/s10278-018-0082-y)
[2] [Open Health Imaging Foundation](https://ohif.org/)


## Dataset  

This Udacity project dataset was adapted from the Medical Decathlon competition "Hippocampus" dataset.  It is stored as a collection of NIFTI files, with one file per image volume and one file per corresponding segmentation mask volume
The original Medical Decathlon "Hippocampus" dataset images are T2 MRI scans of the full brain.  In Udacity's adaptation, the volumes are cropped to only the region around the right hippocampus.  This reduces the dataset size and allows for shorter model training times.

Algorithms that crop rectangular regions of interest are quite common in medical imaging, but deep learning networks are not.

**References**
[1] Amber L. Simpson, Michela Antonelli, Spyridon Bakas, Michel Bilello, Keyvan Farahani, Bram van Ginneken, Annette Kopp-Schneider, Bennett A. Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M. Summers, Patrick Bilic, Patrick F. Christ, Richard K. G. Do, Marc Gollub, Jennifer Golia-Pernicka, Stephan H. Heckers, William R. Jarnagin, Maureen K. McHugo, Sandy Napel, Eugene Vorontsov, Lena Maier-Hein, M. Jorge Cardoso. 
"A large annotated medical image dataset for the development and evaluation of segmentation algorithms," arXiv:1902.09063 (Feb 2019) [LINK](https://arxiv.org/abs/1902.09063)




 to extract & pre-process NIFTI volume data, split data into training and validation sets, load data to PyTorch to train a UNET model, and validate model performance.
1. Data extraction

2. Volume prepation 

3. Volume Slice by Slice

4. PyTorch loading

5. UNET Model training

6.   Why and performance





**Fine Tuning Convolutional Neural Network VGG16 for Pneumonia Detection from X-Rays**  
This project's model was created by fine-tuning ImageNet's VGG16 CNN model with chest X-Ray images.  
To fine-tune the VGG16 model, a new Keras Sequential model was created by taking VGG16 model layers 
and freezing their ImageNet-trained weights.  Subsequent Dense and Dropout layers were added, which will have their weights trained 
for classifying chest X-Ray images for pneumonia.
Model predictions initially return as probabilities between 0 and 1.  These probabilistic results were compared 
against ground truth labels.  A threshold analysis was completed to select the boundary at which 
probalistic results are converted into binary results of either pneumonia presence or absence.
 
The paper of Pranav Rajpurkar et al. (2017), "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning", 
provides a reference to compare against.  This paper established F1-scores as the performance metric to compare radiologists' and algorithms' 
performance in identifying pneumonia in a subset of 420 images from the ChestX-ray14 dataset (Wang et al., 2017). 
F1-scores are the harmonic average of the precision and recall of a model's predictions against ground truth labels.
The CheXNet algorithm achieved an F1 score of 0.435, while a panel of four independent Radiologists averaged an F1 score of 0.387. 
This project's final F1 score is 0.36, which is similar in performance to the panel of Radiologist. 

- For further information about the model architecture, please read the "Algorithm Design and Function" section of 
the [`FDA_Preparation.md`](https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX/blob/master/FDA_Preparation.md).
- Please read [`2_Build_and_Train_Model.ipynb`](https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX/blob/master/2_Build_and_Train_Model.ipynb) for full details of model training and threhold selection.

**References**
[1]  Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, Matthew P. Lungren, Andrew Y. Ng, "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning,"  arXiv:1711.05225, Dec 2017. [Link](https://arxiv.org/abs/1711.05225)   
[2]  Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers.ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017 


**Making Predictions**  
The [`3_Inference Jupyter Notebook`](https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX/blob/master/3_Inference.ipynb)
contains the functions to load DICOM files, pre-process DICOM image, 
load the model built in 2_Build_and_Train_Model, and predict the presence of pneumonia from the DICOM image.

Inputs:
- .dcm DICOM medical imaging file, contains metadata and a medical image

Output:
- DICOM image is displayed with a prediction of whether the patient is Positive or Negative for Pneumonia

The following steps should be performed to analyze a chest X-Ray DICOM file:
1.  Load DICOM file with `check_dicom(filename)` function.  It's output is the DICOM pixel_array or 
an error message if the DICOM file is not a Chest X-Ray.    
2.  Pre-process the loaded DICOM image with `preprocess_image(img=pixel_array, img_mean=0, img_std=1, img_size=(1,224,224,3))` function.
3.  Load trained model with `load_model(model_path, weight_path)`.
4.  Make prediction with `predict_image(model, img, thresh=0.245)`.


### Dataset
The ChestX-ray14 dataset was  curated by Wang et al. and was released by NIH Clinical Center.
It is comprised of 112,120 X-Ray images with disease labels from 30,805 unique patients. 
The disease labels for each image were created using Natural Language Processing (NLP) to process 
associated radiological reports for fourteen common pathologies. The estimated accuracy of the NLP labeling accuracy is estimated to be >90%.

**References**
[1]  Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers.ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017 

## Getting Started

1. Set up your Anaconda environment.  
2. Clone `https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX` GitHub repo to your local machine.
3. Section 1  Open `1_EDA.ipynb` with Jupyter Notebook for exploratory data analysis.
4. Section 2  Open `2_Build_and_Train_Model.ipynb` with Jupyter Notebook for image pre-processing with keras ImageDataGenerator, 
ImageNet VGG16 CNN model fine-tuning, and threshold analysis.
5. Section 3  Open `3_Inference.ipynb` with Jupyter Notebook for inference with a DICOM file.
6. Complete Project Discussion can be found in `FDA_Preparation.md`

### Dependencies
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

**Create local environment**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
**CHANGE**
```
git clone https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX.git
cd Pneumonia_Detection_ChestX
```

2. Create (and activate) a new environment, named `udacity-ehr-env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n udacity-ehr-env python=3.7
	source activate udacity-ehr-env
	```
	- __Windows__: 
	```
	conda create --name udacity-ehr-env python=3.7
	activate udacity-ehr-env
	```
	
	At this point your command line should look something like: `(udacity-ehr-env) <User>:USER_DIR <user>$`. The `(udacity-ehr-env)` indicates that your environment has been activated, and you can proceed with further package installations.



6. Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
 
```
pip install -r pkgs.txt
```


## Project Instructions
please read Udacity's original project instructions in the `Project_Overview.md` file.

**Project Overview**

   1. Exploratory Data Analysis
   2. Building and Training Your Model
   3. Clinical Workflow Integration
   4. FDA Preparation


## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md)
