# Automated Segmentation of Deep Brain Nuclei using Convolutional Neural Networks and Susceptibility Weighted Imaging

## Background
The current repository contains the code used to train and evaluate the segmentation framework (SWI-CNN) presented in the paper "Automated Segmentation of Deep Brain Nuclei using Convolutional Neural Networks and Susceptibility Weighted Imaging". This framework implements five different U-Net architectures (3D U-Net, V-Net, U-Net++, FC-Dense Net, and Dilated FC-Dense Net) as well as an ensemble of these models (EMMA) to perform the segmentation of the dentate nucleus (DEN), red nucleus (RN), substantia nigra (SN), and the subthalamic nuclei (STN) from SW images. It uses a two steps approach to 1) localize the regions on lower resolution images, and 2) segment the regions from full resolution images. This framework allows for the prediction of individual regions as well as the prediction of multiple anatomically close regions (RN, SN, STN).

## Installation
1. Clone this repository. Download the trained models (from https://download.i-med.ac.at/neuro/archive/swi-cnn_models.tar.gz) and extract them in the main directory.
2.  Install Python 3 and the following dependencies:
```
python=3.7.5
tensorflow=1.15.0
keras=2.2.4
numpy=1.17.4
scipy=1.3.2
nibabel=2.5.1
scikit-learn=0.21.3
skimage=0.15.0
pydot=1.4.1
antspy=0.2.2
```
Using Anaconda (https://www.anaconda.com/) this step can be performed with the following commands:

```
conda create -n swi-cnn python=3.7.5 tensorflow-gpu=1.15.0 keras=2.2.4 numpy=1.17.4 scipy=1.3.2 scikit-learn=0.21.3 pydot=1.4.1
conda activate swi-cnn
conda install -c conda-forge scikit-image=0.15.0
pip install antspyx==0.2.2
git clone https://github.com/deepmind/surface-distance.git
pip install surface-distance/
```

Different versions of the packages should not significantly alter the framework, nonetheless, we report here what versions were used for reproducibility. We note that the ```tensowflow=1.15.0``` dependency was critical for us to enable large models with our Nvidia Titan V GPU. The current approach did not work with ```tensorflow=2.0```.

3.  Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) (required for pre-processing)

4. Add the fullpath of the ```swi-cnn```  to your ```PATH``` envorinment variable, i.e. by adding
```
export PATH=$PATH:<fullpath>/swi-cnn
```
to your .basrch file.

## Labeling new datasets

### 1. Pre-processing

Pre-processing of the SW images (i.e. resampling to isotropic resolution, brain mask creation, N4 bias field correction, and SRS normalization) is performed using the following command:
```
swi-cnn.py -s example_subject --preproc
```
The preprocessde images are stored in the ```preproc``` directory. The normalized SW image is named ```preproc/example_subject_brain_norm.nii.gz```.

### 2. Labeling
Labeling of the pre-processed SW images is performed with the command
```
swi-cnn.py --s example_subject --label_regions
```
This will perform the patch selection and subsequent region labeling for each region. The final labels for each model are stored in the corresponding ```labels/<model>``` directory.

## Training of new models

Although this segmentation framework was developed specifically for labeling deep brain nuclei in SW images, new models can easily be trained to segment any region sufficiently small from any type of images given some ground truth labels

### 1. SW images & Ground truth labels

SW images (or any other modality) are stored in the ```swi``` directory. For each ```subject```, a corresponding SW image ```swi/subject.nii.gz``` and matching ground truth labels ```manual_labels/subject.nii.gz``` files are expected.

### 3. Pre-processing
Pre-processing of images is performed using the command
```
swi-cnn.py --subject_list list --preproc
```
where ```list``` is a text file with one subject per line.

The isotropic resolution for resampling (0.6875 mm by default) can be specified using the flag ```--iso_resolution```.

### 4. Augmentation & Downsampling

Off-line augmentation of the training data and subsequent downsampling is performed using
```
swi-cnn.py --subject_list list --augment --downsampling
```
The augmented and downsampled data is stored in the ```augment``` and ```downsampled``` directories, respectively. Here the ```--n_jobs``` flag can be used to specify the number of parallel processes to run for data augmentation and downsampling.

### 5. Training & Evaluation of the Patch Extraction Models

Preallocation of the batches used for training of the Patch Extraction models is performed using
```
swi-cnn.py --subject_list list --preallocate_patch
```
For evaluation purposes, cross-validation of the Patch extraction models is ran using
```
swi-cnn.py --subject_list list --patch_cv
```
Training of the final Patch Extraction models using all available data is performed with
```
swi-cnn.py --subject_list list --patch_final
```
All patch extraction models are stored in the ```models/patch``` directory.

### 6. Training & Evaluation of the Segmentation Models

Similarly to the Patch Extraction models, preallocation of the (segmentation) batches, cross-validation and training of the final Segmentation Model is performed with
```
swi-cnn.py --subject_list list --preallocate_segmentation --segmentation_cv -- segmentation_final
```
Cross-validation of the ensemble model is done with
```
swi-cnn.py --subject_list list --ensemble_cv
```
All segmentation models (including all folds and final models) and corresponding labeling of the test data are stored in the ```models/region``` directory.

### Note:
Although not ideal, some parameters are hard coded and must be changed directly in the code. This includes, among others, the number of folds for cross-validation, the number of epochs, the batch size, the initial learning rate, and the step decay for training. This behavior will be improved in subsequent iterations of the code.

## Citations

Please cite the following papers if you end up using the code or models in this repository:

SWI-CNN:
* Beliveau, V., Nørgaard, M., Birkl, C., Seppi, K., & Scherfler, C. (2021). Automated segmentation of deep brain nuclei using convolutional neural networks and susceptibility weighted imaging, Human Brain Mapping, Vol. 42, No. 15, pp. 4809-4822. https://doi.org/10.1002/hbm.25604

3D Unet:
* Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 424–432. https://doi.org/10.1007/978-3-319-46723-8_49

V-Net:
* Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 2016 Fourth International Conference on 3D Vision (3DV), 565–571. https://doi.org/10.1109/3DV.2016.79

U-Net++:
 * Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2019). UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation. IEEE Transactions on Medical Imaging, 1–1. https://doi.org/10.1109/TMI.2019.2959609

FC-Dense Net:
* Jegou, S., Drozdzal, M., Vazquez, D., Romero, A., & Bengio, Y. (2017). The one hundred layers tiramisu: Fully convolutional DenseNets for semantic segmentation. Proceedings of IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 1175–1183. https://doi.org/10.1109/CVPRW.2017.156

Dilated FC-Dense Net:
* Kim, J., Patriat, R., Kaplan, J., Solomon, O., & Harel, N. (2020). Deep cerebellar nuclei segmentation via semi-supervised deep context-aware learning from 7T diffusion MRI. IEEE Access, 8, 101550–101568. https://doi.org/10.1109/ACCESS.2020.2998537

EMMA:
* Kamnitsas, K., Bai, W., Ferrante, E., McDonagh, S., Sinclair, M., Pawlowski, N., … Glocker, B. (2018). Ensembles of multiple models and architectures for robust brain tumour segmentation. Lecture Notes in Computer Science, 10670 LNCS, 450–462. https://doi.org/10.1007/978-3-319-75238-9_38
