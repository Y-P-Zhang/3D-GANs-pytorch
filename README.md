# 3D-GANs-pytorch

A  PyTorch implementation of 3D GANs

## Introduction

**In this implementation aims to build general framework for 3D-GANs using Pytorch, I have implemented or will implement several Genertators,Discriminators and losses.**  

The basics model is SRGAN, which is a super resolution network originally publish in  [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)  

## Train and Test

You can used `python pretrain.py` to pretrain the genertator. This file can also use to do super resolution without GAN.  

run `python train.py` and `python test.py`  
The adjustable parameter is write in `./options`  
You must specify ``--img_width`` ``--img_height`` ``--img_depth`` ``--file_extension`` according to your dataset.   

## Train your own datasets

The folder structure is below:  

3D-GANs-pytorch   
├── datasets    
├  ├── train   
├  ├── test   
├── option.py   
├── train.py   
├── test.py   
├── pretrain.py   
├── models   
├── utils   
├── data  

Your own datasets need to be put into `datasets/train` and `datasets/test`.    

Dataset in this implementation using [Iseg](http://iseg2019.web.unc.edu/),  `nibabel.load(image_file)` is used to load the `image_file`. If your data format is different, you may need to rewrite the `CustomDataset` in `data/customdataset`.     

**NOTE:** You may need to rewrite a normalize method according to the mean and variance of your data set.    

## Completed

### Genertators

* √ ResnetGenertator
* × 3DUnet
* × ResUnet

### Discriminators

* GAN 
* Patch GAN
* Pixel GAN

### Losses

* Gradient difference Loss
* ...

## Example

__High resolution / Low resolution__      
__Genertated with GAN / Genertated with CNN__

![Example](https://github.com/Y-P-Zhang/3D-GANs-pytorch/blob/master/example.png?raw=true)
