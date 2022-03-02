# Per-COVID-19: A Benchmark Dataset for COVID-19 Percentage Estimation from CT-Scans
Covid-19 infection percentage estimation from CT-scans
 
Dataset and Pytorch code for the paper 
[" Per-COVID-19: A Benchmark Dataset for COVID-19 Percentage Estimation from CT-Scans"](https://www.mdpi.com/2313-433X/7/9/189), MDPI, Imaging 2021

[Paper](https://www.mdpi.com/2313-433X/7/9/189) 


# Introduction

COVID-19 infection recognition is a very important step in the fight against the COVID-19 pandemic. In fact, many methods have been used to recognize COVID-19 infection including Reverse Transcription Polymerase Chain Reaction (RT-PCR), X-ray scan, and Computed Tomography scan (CT- scan). In addition to the recognition of the COVID-19 infection, CT scans can provide more important information about the evolution of this disease and its severity. With the extensive number of COVID-19 infections, estimating the COVID-19 percentage can help the intensive care to free up the resuscitation beds for the critical cases and follow other protocol for less severity cases. In this paper, we introduce COVID-19 percentage estimation dataset from CT-scans, where the labeling process was accomplished by two expert radiologists. Moreover, we evaluate the performance of three Convolutional Neural Network (CNN) architectures: ResneXt-50, Densenet-161, and Inception-v3. For the three CNN architectures, we use two loss functions: MSE and Dynamic Huber. In addition, two pretrained scenarios are investigated (ImageNet pretrained models and pretrained models using X-ray data). The evaluated approaches achieved promising results on the estimation of COVID-19 infection. Inception-v3 using Dynamic Huber loss function and pretrained models using X-ray data achieved the best performance for slice-level results: 0.9365, 5.10, and 9.25 for Pearson Correlation coefficient (PC), Mean Absolute Error (MAE), and Root Mean Square Error (RMSE), respectively. On the other hand, the same approach achieved 0.9603, 4.01, and 6.79 for PC_subj, MAE_subj, and RMSE_subj, respectively, for subject-level results. These results prove that using CNN architectures can provide accurate and fast solution to estimate the COVID-19 infection percentage for monitoring the evolution of the patient state. View Full-Text
Keywords: COVID-19; deep learning; convolutional neural network; CT-scans; dataset generation

<p align="center">
  <img src="img/img.png" width="800"/>
</p>

# About This Repository:
This repository contains two main parts:

## 1- Download the Dataset:
The database is available in GoogleDrive at the link: 

https://drive.google.com/drive/folders/1cq11VHn8lTChCDicRi_-xxei1FVftejc?usp=sharing

The slices are available in 'Database1.zip' file, are the labels are available in 'Database\_new.xlsx' file. The columns of the '.xlsx' file represent: the image name, the Covid-19 infection percentage, the fold split and the subject id, respectively.
The above link contains the original dataset, where the lung regions were cropped manually.   

The dataset was enhanced and used for Covid-19-Infection-Percentage-Estimation-Challenge (the original dataset was preprocessed by cropping the lung regions, while in the challenge we provided the original slices images), and more Ct-scans were added.

You can find more details about the challenge here
https://github.com/faresbougourzi/Covid-19-Infection-Percentage-Estimation-Challenge

You can use Codalab to compare your results with other teams (https://competitions.codalab.org/competitions/35575) 

For the challenge there is an extra testing dataset, if you got a result better than the best 10 approaches, you can send us your predictions and we can provide you the testing results

If you have any query,  Please do not hesitate to contact us through: faresbougourzi@gmail.com

## 2- Pytorch Code:

The pytorch code contains three python codes which are: 
- Covid_Per.py: the dataloader function
- Create_database.py: prepare the dataset with traning and validation sets for the 5-folds cross-validation
- train_five_folds_covid_percentage.py: training and testing the five-fold experiment

# Citation:

```bash
@Article{jimaging7090189,
AUTHOR = {Bougourzi, Fares and Distante, Cosimo and Ouafi, Abdelkrim and Dornaika, Fadi and Hadid, Abdenour and Taleb-Ahmed, Abdelmalik},
TITLE = {Per-COVID-19: A Benchmark Dataset for COVID-19 Percentage Estimation from CT-Scans},
JOURNAL = {Journal of Imaging},
VOLUME = {7},
YEAR = {2021},
NUMBER = {9},
ARTICLE-NUMBER = {189},
URL = {https://www.mdpi.com/2313-433X/7/9/189},
ISSN = {2313-433X},
DOI = {10.3390/jimaging7090189}
}
```
```bash
@article{vantaggiato2021covid,
  title={Covid-19 recognition using ensemble-cnns in two new chest x-ray databases},
  author={Vantaggiato, Edoardo and Paladini, Emanuela and Bougourzi, Fares and Distante, Cosimo and Hadid, Abdenour and Taleb-Ahmed, Abdelmalik},
  journal={Sensors},
  volume={21},
  number={5},
  pages={1742},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
If you use the Dynamic Regression Loss functions, Please Cite the original paper:

```bash
@article{bougourzi_deep_2022,
	title = {Deep learning based face beauty prediction via dynamic robust losses and ensemble regression},
	author = {Bougourzi, F. and Dornaika, F. and Taleb-Ahmed, A.},	
	volume = {242},
	issn = {0950-7051},
	url = {https://www.sciencedirect.com/science/article/pii/S0950705122000740},
	doi = {10.1016/j.knosys.2022.108246},
	language = {en},
	urldate = {2022-03-02},
	journal = {Knowledge-Based Systems},
	year = {2022},
	keywords = {Convolutional neural network, Deep learning, Ensemble regression, Facial beauty prediction, Robust loss functions},
	pages = {108246},
}}
```


