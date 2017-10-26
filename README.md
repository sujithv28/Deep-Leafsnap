# Deep-Leafsnap

Convolutional Neural Networks have become largely popular in image tasks such as image classification recently largely due to to Krizhevsky, et al. in their famous paper [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). Famous models such as AlexNet, VGG-16, ResNet-50, etc. have scored state of the art results on image classfication datasets such as ImageNet and CIFAR-10.

We present an application of CNN's to the task of classifying trees by images of their leaves; specifically all 185 types of trees in the United States. This task proves to be difficult for traditional computer vision methods due to the high number of classes, inconsistency in images, and large visual similarity between leaves.

Kumar, et al. developed a automatic visual recognition algorithm in their 2012 paper [Leafsnap: A Computer Vision System for Automatic Plant Species Identification](http://neerajkumar.org/base/papers/nk_eccv2012_leafsnap.pdf) to attempt to solve this problem.

Another goal of our Convolutional networks is to be able to run on the LeafSnap mobile app. There has been a number of recent research efforts to develop networks that are capable of running on compute-constrained devices, one such effort is [MobileNet](https://arxiv.org/abs/1704.04861). MobileNet has tunable hyperparameters that allow the network to be reduced to different sizes depending on just how constrained your resources are. The full version (MobileNet 1.0) has comparable accuracy to VGG16 and GoogLeNet, but with a drastic reduction in parameters and compute time at inference. We now experiment with a MobileNet of different sizes on our goal task, and compare its accuracy and speed with other models. 

Our first model is based off VGG-16 except modified to work with `64x64` size inputs due to computational constraints. We achieved state of the art results at the time. Our deep learning approach to this problem further improves the accuracy from `70.8%` to `86.2%` for the top-1 prediction accuracy and from `96.8%` to `98.4%` for top-5 prediction accuracy.

|                      | Top-1 Accuracy | Top-5 Accuracy |
|----------------------|:--------------:|:--------------:|
|       Leafsnap       |      70.8%     |      96.8%     |
| Deep-Leafsnap VGG-16 |      86.2%     |      98.4%     |

We noticed that our model failed to recognize specific classes of trees constantly causing our overall accuracy to derease. This is primarily due to the fact that those trees had very small leaves which were hard to preprocess and crop. Our training images were also resized to `64x64` due to limited computational resources. 

Our second model we trained was a MobileNet using the full 224x224 images. Despite the increase in image size, the MobileNet still uses significantly less compute and number of parameters than the constrained VGG-16 network. After only 10 epochs of training with no pretraining, the results were as follows.


|               | Top-1 Accuracy | Top-5 Accuracy | Batch Images / Sec | Single Image / Sec |
|---------------|:--------------:|:--------------:|:------------------:|:------------------:|
| MobileNet 1.0 |      93.4%     |      99.3%     |                    |                    |  
| MobileNet .25 |      xx.x%     |      xx.x%     |                    |                    |  
|     VGG-16    |      yy.y%     |      yy.y%     |                    |                    |

Due to the fact that MobileNet provides substantial reductions in parameter count, the resulting mobile app will have greater speed performance, use less battery power, and have less memory footprint. 

The following goes over the code and how to set it up on your own machine.

## Files
* `model.py` trains a convolutional neural network on the dataset.
* `vgg.py` PyTorch model code for VGG-16.
* `densenet.py` PyTorch model code for DenseNet-121.
* `resnet.py` PyTorch model code for ResNet.
* `mobilenet.py` PyTorch model code for MobileNet.
* `dataset.py` creates a new train/test dataset by cropping the leaf and augmenting the data.
* `utils.py` helps do some of the hardcore image processing in dataset.py.
* `averagemeter.py` helper class which keeps track of a bunch of averages when training.
* `leafsnap-dataset-images.csv` is the CSV file corresponding to the dataset.
* `requirements.txt` contains the pip requirements to run the code.
* `setup_scriptGPU.sh` is a script to install the requirements and download the dataset.
* `setup_script.sh` does the same as the GPU script, but for non GPU enabled machines.

## Installation
To run the models and code make sure you have [Python](https://www.python.org/downloads/) and Pip installed.

Clone the repo onto your local machine and cd into the directory.
```
git clone https://github.com/sujithv28/Deep-Leafsnap.git
cd Deep-Leafsnap
```

If you are running a machine with GPU capabilities, run the following command:
```
bash setup_scriptGPU.sh
```
This will install the necessary requirements, install Torch with CUDA 8.0, and download the dataset.

If you do not have a GPU machine, run:
```
bash setup_script.sh
```

Check to make sure the OpenCV installation was succesful. You can do this by running and making sure nothing complains:
```
python
import cv2
```

## Create the Training and Testing Data
To create the dataset, run
```
python dataset.py
```
This cleans the dataset by cropping only neccesary portions of the images containing the leaves and also resizes them to `64x64`. If you want to change the image size go to `utils.py` and change `img = misc.imresize(img, (64,64))`to any size you want.

## Training Model
To train the model, run
```
python model.py
```
