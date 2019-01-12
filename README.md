# EDSR-PP
SuperResolution

### Introduction

In this article, we propose efficient module based single image SR networks
(EMBSR) and tackle multiple SR problems in NTIRE 2018 SR challenge by recycling trained networks. Our proposed
EMBSR allowed us to reduce training time with effectively deeper networks, to use modular ensemble for improved
performance, and to separate subproblems for better performance. We also proposed EDSR-PP, an improved version
of previous ESDR by incorporating pyramid pooling so that global as well as local context information can be
utilized. Our proposed EDSR-PP achieved honorable mentina award for Track1: classic Bicubic X8 in NTIRE 2018 SR challenge.

### Citations

If you are using the code/model/data provided here in a publication, please consider citing our paper:

    @inproceedings{park2018efficient,
      title={Efficient module based single image super resolution for multiple problems},
      author={Park, Dongwon and Kim, Kwanyoung and Chun, Se Young},
      booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      pages={995-1003},
      year={2018}
    }  
    
    
### Installation

Install <a href="https://pytorch.org/">pytorch</a>. The code is tested under 0.4.1 GPU version and Python 3.6  on Ubuntu 16.04.

### Training RCF

1. Download the datasets you need.

2. Start training process by running following commands:

    ```Shell
    sh demo.sh
    ```
    
### Acknowledgment

This code is based on EDSR. Thanks to the contributors of EDSR.

    @inproceedings{lim2017enhanced,
      title={Enhanced deep residual networks for single image super-resolution},
      author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
      booktitle={The IEEE conference on computer vision and pattern recognition (CVPR) workshops},
      pages={1132-1140},
      year={2017}
    }
