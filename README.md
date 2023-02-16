# Quantifying Alzheimer's Disease Progression Through Automated Measurement of Hippocampal Volume
This repository contains a completed cap-stone project for the Udacity "Applying AI to 3D Medical Imaging Data" course, 
part of the AI for Healthcare Nanodegree program.  It has been reviewed by Udacity instructors and met project specifications.

# Table of Contents
- [Introduction](#introduction)
	- [Dataset](#dataset)
- [Getting Started](#getting-started)
	- [Installation](#1-installation)
	- [Create and Activate the Environment](#2-create-and-activate-the-environment)
- [Project Instructions](#project-instructions)
	- [Part 1: Curating a Dataset for Machine Learning Training and Validation](#part-1-curating-a-dataset-for-machine-learning-training-and-validation)
	- [Part 2: Train U-Net Fully Convolutional Network for Brain Segmentation](#part-2-train-u-net-fully-convolutional-network-for-brain-segmentation)
	- [Part 3: Simulate Integration of Segmentation CNN into DIMSE](#part-3-simulate-integration-of-segmentation-cnn-into-dimse)
- [License](#license) 

# Introduction  

Alzheimer's Disease (AD) is a degenerative brain disease that affects an estimated 5.8 million Americans age 65 and older in 2020.
It is thought that AD begins 20 years or more before symptoms arise, with progressive brain changes that are unnoticeable to the affected person.  As the disease progresses, nerve cells (neurons) in parts of the brain involved with thinking, learning, and memory functions are damaged and destroyed.  
After years of brain changes, individuals experience symptoms such as memory loss, loss of language function, and other manifestations.  AD is the most common cause of dementia [1].

The Alzheimer's Association (AA) "2020 Alzheimer's Disease Facts and Figures" estimates that the number of Americans with AD may triple by 2050 [1].
With such a staggering future care need, projections show that there will be a shortage of front-line primary care physicians (PCP), neurologists, and other specialists who provide critical expertise in dementia diagnosis and care [2].

Currently, an MRI exam is one of the most advanced methods to quantify AD.  Studies have shown that measurements of hippocampal volume from MRI exams is useful to diagnose and track progression of several brain diseases, including AD.  AD patients have shown a reduced hippocampus volume.
Quantifying disease progression over time can help direct therapy and disease management. However, the process to measure the hippocampus using MRI scans is very time consuming.  Each 3D MRI scan volume contains several dozen 2D images slices.  With each 2D image slice, the hippocampus must be correctly identified and traced.

AI software can provide a practical solution to quantify hippocampal volume from MRI scans.  Deep learning algorithms for computer vision segmentation tasks introduce new avenues to automate the identification of objects and trace objects in an image.   
For this project, a deep learning segmentation model was created to identify hippocampus structures in brain MRI scans on volume pixel (voxel) level.  The identified hippocampus voxels are translated to physical volume measurements in mm^3.

The intention of this software is to be integrated into a Picture Archiving and Communication System (PACS) whereby this software will automatically calculate hippocampal volumes of new MRI studies as the studies are committed to a clinical imaging archive server.
This software will eliminate the tedious hippocampus measurement task from physicians' workflow and will quickly provide physicians with an accurate measurement.  The software will also provide a consistent method to trace the hippocampus structure, whereas there may be variability between clinicians in the tracement task.
The performance metrics requirements for this segmentation CNN are to achieve Dice Similarity Coefficient >0.90 and Jaccard Index >0.80 when comparing model predictions to ground truth segmentation masks.  

 ![report.dcm](/Section%203%20Simulate%20DIMSE/out/Study1_DCM%20Report%20Screenshot.jpg)  
 **Figure 1.** Example report output for Test Volumes Study 1

This project is broken into three sections and are located in separate folders:
- Section 1 Curating a Dataset of Brain MRIs: Analyze Medical Segmentation Decathlon dataset metadata, analyze & visualize image volumes & corresponding labels, and identify & remove data that is not of a brain MRI.  
- Section 2 Training a segmentation CNN model: Image volume extraction from NIFTI files, image volume pre-processing, split dataset using Scikit-Learn, build & train a UNet Fully Convolutional Neural Network (FCN) with PyTorch, 
and evaluate model performance metrics - overall Dice Similarity Coefficient & Jaccard Index.  
- Section 3 Integrating into a Clinical Network:  Simulate DICOM Message Service Element (DIMSE). A dedicated AI computer will be added to a clinical PACS network.  The AI computer will contain a copy of the Section 2 segmentation CNN.  When a MRI scanner completes a scan and sends a MRI study to the PACS, the AI computer will receive a copy of the transferred file to execute inference and provide a DICOM report with hippocampus measurements.

In this completed model run, the model achieved performance of **Overall Mean Dice Similarity Coefficient 0.906** and **Overall mean Jaccard Index 0.830**.  A full discussion of completed project results and model performance can be read in [Validation_Plan_Proposal](Validation_Plan_Proposal.pdf)  

**References**  
[1] Alzheimer’s Association. "2020 Alzheimer’s Disease Facts and Figures", Alzheimers & Dementia, 2020;16(3):391+. [LINK](https://www.alz.org/media/Documents/alzheimers-facts-and-figures_1.pdf)  
[2] "Primary Care Physicians on the Front Lines of Diagnosing and Providing Alzheimer’s and Dementia Care: Half Say Medical Profession Not Prepared to Meet Expected Increase in Demands". www.alz.org, 2020 [LINK](https://www.alz.org/news/2020/primary-care-physicians-on-the-front-lines-of-diag)


## Dataset  

The project dataset was provided by Udacity. It was adapted from the Medical Segmentation Decathlon "Hippocampus" dataset. The original "Hippocampus" dataset consisted of cropped T2 MRI scans of the full brain.  The volumes were cropped to only the region around the right hippocampus.  This reduces the dataset size and allows for shorter model training times.
The project dataset was stored as a collection of NIFTI files, with one file per image volume and one file per corresponding segmentation mask volume

**NOTE** Udacity's project dataset is not provided in this GitHub repo, as it is not a public dataset.  Please enroll in the Udacity AI for Healthcare Nanodegree to access a copy of the dataset.

**References**  
[1] Amber L. Simpson, Michela Antonelli, Spyridon Bakas, Michel Bilello, Keyvan Farahani, Bram van Ginneken, Annette Kopp-Schneider, Bennett A. Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M. Summers, Patrick Bilic, Patrick F. Christ, Richard K. G. Do, Marc Gollub, Jennifer Golia-Pernicka, Stephan H. Heckers, William R. Jarnagin, Maureen K. McHugo, Sandy Napel, Eugene Vorontsov, Lena Maier-Hein, M. Jorge Cardoso. 
"A large annotated medical image dataset for the development and evaluation of segmentation algorithms," arXiv:1902.09063 (Feb 2019) [LINK](https://arxiv.org/abs/1902.09063)


# Getting Started

1. Set up your Anaconda environment.  
2. Clone `https://github.com/ElliotY-ML/Hippocampus_Segmentation_MRI.git` GitHub repo to your local machine.
3. Section 1:  Open a Jupyter Notebook.  Navigate to directory `Section 1 EDA` and open `Final Project EDA.ipynb` for exploratory data analysis.  See the Project Instructions section of this README for further instructions.
4. Section 2:  To train a Hippocampus Segmentation CNN, follow the instructions provided in the Project Instructions section of this README.  
	To explore the modules that `run_pipeline_ml.py` relies on, Open a Python IDE such as Spyder. Open the following Python modules in the Python IDE: 
	- Two modules are contained in `Section 2 Train_Eval_Model/src/data_prep`: 
		1. `HippocampusDatasetLoader.py` contains the function to extract image volume from NIFTI, normalize the image volume, and reshape the image volume into a common volume size. 
	 	2. `SlicesDataset.py` contains the function to numerate all individual images slices belonging to an image volume.  It returns a dictionary containing a slice identifier, MRI scan slice, and corresponding segmentation mask slice. 
	- The `Section 2 Train_Eval_Model/src/networks/RecursiveUNet.py` contains the U-Net architecture.
	- Two modules are contained in `Section 2 Train_Eval_Model/src/utils`: 
		1. `volume_stats.py` contains the functions to compute the Dice Similarity Coefficients for two 3-D volumes and the Jaccard Index. 
	 	2. `utils.py` contains the functions to plot an array of images, log data to TensorBoard, save numpy as an image, and pad image volumes to a specified shape.
	- The `Section 2 Train_Eval_Model/src/experiments/UNetExperiment.py` contains the functions to load training and validation data batches to PyTorch, train the U-Net model, log training to TensorBoard, save model parameters, run validation, and compute performance metrics.  
	- The `Section 2 Train_Eval_Model/src/inference/UNetInferenceAgent.py` contains functions for single volume inference and returns a prediction mask.
		
5. Section 3:  Modules in this section should be explored with a Python IDE.  Follow the instructions provided in the Project Instructions section of this README to setup a DIMSE simulation and run inference on MRI studies.
6. Complete project results discussion can be found in `Validation_Plan_Proposal.pdf`

## Dependencies
Using Anaconda consists of the following:

1. Install [`anaconda`](https://www.anaconda.com/products/individual) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

### 1. Installation

**Download** the latest version of `anaconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86_64.exe
[win32]: https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86.exe
[mac64]: https://repo.anaconda.com/archive/Anaconda3-2020.11-MacOSX-x86_64.sh
[lin64]: https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
[lin32]: https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86.sh

**Install** [anaconda](https://docs.anaconda.com/anaconda/) on your machine. Detailed instructions:

- **Linux:** https://docs.anaconda.com/anaconda/install/linux/
- **Mac:** https://docs.anaconda.com/anaconda/install/mac-os/
- **Windows:** https://docs.anaconda.com/anaconda/install/windows/

### 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

### Git and version control
These instructions also assume you have `git` installed for working with GitHub from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

**Create local environment**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.

```
git clone https://github.com/ElliotY-ML/Hippocampus_Segmentation_MRI.git
cd Hippocampus_Segmentation_MRI
```

2. Create and activate a new environment, named `hippo-segmentation` with Python 3.7+.  Be sure to run the command from the project root directory since the environment.yml file is there.  If prompted to proceed with the install `(Proceed [y]/n)` type y and press `ENTER`.

	- __Linux__ or __Mac__: 
	```
	conda env create -f environment.yml
	source activate hippo-segmentation
	```
	- __Windows__: 
	```
	conda env create -f environment.yml
	conda activate hippo-segmentation
	```
	
	At this point your command line should look something like: `(hippo-segmentation) <User>:USER_DIR <user>$`. The `(hippo-segmentation)` indicates that your environment has been activated.


**In the 3rd section of the project we will be working with three software products for emulating the clinical network.**  

You would need to install and configure:
1. [Orthanc server](https://www.orthanc-server.com/download.php) for PACS emulation
2. [OHIF zero-footprint web viewer for viewing images](https://docs.ohif.org/development/getting-started.html). Note that if you deploy OHIF from its GitHub repository, at the moment of writing the repo includes a yarn script `orthanc:up` where it downloads and runs the Orthanc server from a Docker container. If that works for you, you won't need to install Orthanc separately.
3. If you are using Orthanc (or other DICOMWeb server), you will need to configure OHIF to read data from your server. OHIF has instructions for this: https://docs.ohif.org/configuring/data-source.html
4. In order to fully emulate the Udacity workspace, you will also need to configure Orthanc for auto-routing of studies to automatically direct them to your AI algorithm. For this you will need to take the script that you can find at `section3/src/deploy_scripts/route_dicoms.lua` and install it to Orthanc as explained on this page: https://book.orthanc-server.com/users/lua.html
5. [DCMTK tools](https://dcmtk.org/) for testing and emulating a modality. Note that if you are running a Linux distribution, you might be able to install dcmtk directly from the package manager (e.g. `apt-get install dcmtk` in Ubuntu)



# Project Instructions

The original Udacity project instructions can be read in the [`Udacity_Project_Instructions.md`](Udacity_Project_Instructions.md) file.

**Project Overview**

   1. Exploratory Data Analysis and Curating a Dataset
   2. Train U-Net Fully Convolutional Network for Brain Segmentation
   3. Simulate Integration of Segmentation CNN into Clinical DIMSE
   4. Validation Plan Proposal


## Part 1: Curating a Dataset for Machine Learning Training and Validation

The human brain has two hippocampi, one in the left hemisphere and one in the right hemisphere of the brain.  Udacity provided this project's dataset that consists of cropped regions around the right hippocampus.
The dataset may also contain MRI scan volumes of other anatomies.  This Section of the project reviews the given dataset to clean the dataset, and retrieve only Brain MRI scan volumes.

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


## Part 2: Train U-Net Fully Convolutional Network for Brain Segmentation

In Section 2, PyTorch is used for training a model with the U-Net convolutional neural network architecture from the University of Freiburg [1] for segmentation of Brain MRIs and identify the right hippocampus.  
Cleaned data from Section 1 is the input into Section 2.   The directory `/Section 2 Train_Eval_Model/src` contains the source code that forms the machine learning pipeline.  

Inputs:
- `/Section 2 Train_Eval_Model/images` contains 260 NIFTI Files containing cropped Brain MRI volumes 
- `/Section 2 Train_Eval_Model/labels` contains 260 NIFTI Files containing Right Hippocampus Labels 
 
Outputs:  
*Stored in `/Section 2 Train_Eval_Model/out` in folders named "YYYY-MM-DD_Basic-unet":
- Trained model and weights for segmentation of Hippocampus in brain MRI volumes stored in file named `model.pth`.
- Model performance metrics information, Dice Similarity Coefficient and Jaccard Index, stored in `results.json` file.

Instructions:  
1. Open a Terminal and Run script `/Section 2 Train_Eval_Model/src/run_ml_pipeline.py`.  It will call and execute methods from modules contained in the `/src/` tree to extract & pre-process NIFTI Brain MRI volumes, complete model training, and evaluate performance.
2. `run_ml_pipeline.py` has hooks to log progress to Tensorboard.  To see the Tensorboard output, launch Tensorboard executable from the same directory where `run_ml_pipeline.py` is location by using the command:
> tensorboard --logdir runs --bind_all
3.  Tensorboard will write logs into the director called `runs`.  View the progress by opening a browser and navigate to port 6006 of the machine where you are running it.

In a completed model run, the model achieved performance of **Overall Mean Dice Similarity Coefficient 0.906** and **Overall mean Jaccard Index 0.830**.  This meets requirements for Dice Similarity Coefficient >0.90 and Jaccard Index >0.80.

**References**  
[1]  Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015, available at arXiv:1505.04597 [cs.CV] 


## Part 3: Simulate Integration of Segmentation CNN into DIMSE

In Section 3, the segmentation CNN from Section 2 will be integrated into a simulated clinical network.  This AI product will automatically compute hippocampus volume for brain MRI scans, and provide this information to clinicians in a DICOM report.  

![Clinical Network Setup](/data/readme.img/network_setup.png)  
**Figure 2.** DIMSE Simulation Setup  

List | Network Object	| Script to Simulate Network Object
--- | --- | ---
1 | Picture Archiving & Communications System (PACS) server | Orthanc DICOM server [1]
2 | MRI Scanner 											| `section3/src/deploy_scripts/send_volume.sh`.  It will initiate a file transfer to the Orthanc.
3 | Viewer System 											| OHIF Viewer [2].  It connects to the Orthanc server using DicomWeb and is serving a web application on port 3000. 
4 | AI Server containing Segmentation software				| (1) `section3/src/deploy_scripts/start_listener.sh`.  It will copy everything it receives into a folder specified in the script.<br>(2) `Section 3 Simulate DIMSE/src/inference.py` is the Hippocampus Segmentation CNN software.

1.  The PACS server is central to clinical settings.  It receives & archives all medical images and allows connected computers to request & send image files.  The Orthanc software, by Sébastien Jodogne, is a standalone DICOM server which allows the simulation of a PACS server [1].
 For this project, the Orthanc is listening to DICOM DIMSE requests on port 4242 and has a DicomWeb interface that is open at port 8042.  It is also running a model that sends everything it receives to an AI server.
2.  The MRI Scanner will send entire studies to the Picture Archiving and Communication System (PACS) Orthanc server after completing a scan.  The script will simulate the archive transfer.
3.  The Viewer system represents workstations that clinicians use to retrieve and view studies from PACS.  The OHIF is viewer is software for viewing medical studies.  It is connecting to the Orthanc server using DicomWeb and is serving a web application on port 3000.
4.  An AI server is responsible for listening to PACS ports for incoming MRI studies.  When it detects that an MRI study is sent, the AI server will request a copy from the PACS server.  Once the MRI study is received on the AI server, the brain MRI scan will be processed by segmentation software and the hippocampus volume will be calculated from the determined hippocampus mask.  

Inputs:  
- A file transfer of a Brain MRI scan.

Outputs:
- A DICOM Report displaying Total Hippocampal Volume, Anterior Hippocampal Volume, Posterior Hippocampal Volume, and Axial views (head to toe direction) at three depths.


Instructions:

1.  Copy Trained segmentation model `model.pth` from Section 2 into folder `/Section 3 Simulate DIMSE/src/inference`.
2.  Set up Orthanc by opening a terminal and enter the following:
`bash launch_orthanc.sh` or `./launch_orthanc.sh`. Don't close this terminal.  
Wait for it to complete, with the last line being something like
`W0509 05:38:21.152402 main.cpp:719] Orthanc has started` and/or you can verify that Orthanc is working by running `echoscu 127.0.0.1 4242 -v` in a new terminal.
3.  Set up OHIF.  Open a new terminal and enter the following
`bash launch_OHIF.sh` or `./launch_OHIF.sh`. Don't close this terminal
Wait for it to complete, with the last line being something like
`@ohif/viewer: ℹ ｢wdm｣: Compiled with warnings.`  
You will then want to enter the Desktop with the bottom right hand corner.
-  OHIF should automatically open in a Web Browser but if not you can paste `localhost:3005` into the address bar of a Web browser window.
-  orthanc isn't necessary to open but if you need it you can access it can paste `localhost:8042` into the address bar of a Web browser window.
4. Open a terminal and cd to `Section 3 Simulate DIMSE/src`.  Run `start_listener.sh`.  Keep this terminal open.
5. Edit `/Section 3 Simulate DIMSE/src/deploy_scripts/send_volume.sh` to specify target MRI study, such as `storescu 127.0.0.1 4242 -v -aec HIPPOAI +r +sd /data/TestVolumes/Study1`
6. Open another terminal for simulating MRI transfer from MRI scanner to PACS. cd to `Section 3 Simulating DIMSE/src` and run `send_volume.sh`.  A copy of the specified MRI study in step 5 will be added to `Section 3 Simulate DIMSE/src/data/TestVolumes/`
7. Open another terminal to execute Hippocampus Segmentation program.  cd to `Section 3 Simulate DIMSE/src`.  Run `inference.py ../../data/TestVolumes/StudyName`, where the `../../data/TestVolumes/StudyName` folder contains a folder with DICOM files belonging to one brain MRI study.  
8. The output is a DICOM report, `datetime_report.dcm`, and three cross-sectional `.png` images of the brain MRI with highlighted hippocampus structures stored in `Section 3 Simulate DIMSE/out` and the report is automatically stored to the Orthanc. 
9. The output `Section 3 Simulate DIMSE/out/datetime_report.dcm` can be viewed with OHIF in a web browser.    
   
![report.dcm](/Section%203%20Simulate%20DIMSE/out/Study2_DCM%20Report%20Screenshot.jpg)  
**Figure 3.** Example report for Test Volumes Study2  

 ![report.dcm](/Section%203%20Simulate%20DIMSE/out/Study3_DCM%20Report%20Screenshot.jpg)  
**Figure 4.** Example report for Test Volumes Study3  

**References**  
[1] Jodogne, S. The Orthanc Ecosystem for Medical Imaging. Journal of Digital Imaging 31, 341–352 (2018). [Link](https://doi.org/10.1007/s10278-018-0082-y)  
[2] [Open Health Imaging Foundation](https://ohif.org/)



# License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md)
