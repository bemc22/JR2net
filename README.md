# JR2net: A Joint Representation and Recovery Network for Compressive Spectral Imaging

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/JR2net/blob/main/demo_inference.ipynb)
[![DOI:10.1364/AO.463726](https://zenodo.org/badge/DOI/10.1364/AO.463726.svg)](https://doi.org/10.1364/AO.463726)
[![arXiv](https://img.shields.io/badge/arXiv-2205.07770-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2205.07770)

## Abstract

Deep learning models are state-of-the-art in compressive spectral imaging (CSI) recovery. These methods use a deep neural network (DNN) as an image generator to learn non-linear mapping from compressed measurements to the spectral image. For instance, the deep spectral prior approach uses a convolutional autoencoder network (CAE) in the optimization algorithm to recover the spectral image by using a non-linear representation. However, the CAE training is detached from the recovery problem, which does not guarantee optimal representation of the spectral images for the CSI problem. This work proposes a joint non-linear representation and recovery network (JR2net), linking the representation and recovery task into a single optimization problem. JR2net consists of an optimization-inspired network following an ADMM formulation that learns a non-linear low-dimensional representation and simultaneously performs the spectral image recovery, trained via the end-to-end approach. Experimental results show the superiority of the proposed method with improvements up to 2.57 dB in PSNR and performance around 2000 times faster than state-of-the-art methods.


## How to cite
If this code is useful for your and you use it in an academic work, please consider citing this paper as


```bib
@article{monroy2022jr2net,
author = {Monroy, Brayan and Bacca, Jorge and Arguello, Henry},
journal = {Appl. Opt.},
keywords = {Electromagnetic radiation; Imaging systems; Light intensity; Medical imaging; Neural networks; Spectral imaging},
number = {26},
pages = {7757--7766},
publisher = {Optica Publishing Group},
title = {JR2net: a joint non-linear representation and recovery network for compressive spectral imaging},
volume = {61},
month = {Sep},
year = {2022},
url = {http://opg.optica.org/ao/abstract.cfm?URI=ao-61-26-7757},
doi = {10.1364/AO.463726},
```
