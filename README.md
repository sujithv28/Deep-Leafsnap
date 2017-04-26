# Deep-Leafsnap

Convolutional Neural Networks have become largely popular in image tasks such as image classification recently largely due to to Krizhevsky, et al. in their famous paper [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). Famous models such as AlexNet, VGG-16, ResNet-50, etc. have scored state of the art results on image classfication datasets such as ImageNet and CIFAR-10.

We present an application of CNN's to the task of classifying trees by images of their leaves; specifically all 185 types of trees in the United States. This task proves to be difficult for traditional computer vision methods due to the high number of classes, inconsistency in images, and large visual similarity between leaves.

Kumar, et al. developed a automatic visual recognition algorithm in their 2012 paper [Leafsnap: A Computer Vision System for Automatic Plant Species Identification](http://neerajkumar.org/base/papers/nk_eccv2012_leafsnap.pdf) to attempt to solve this problem.

Our model is based off VGG-16 except modified to work with `64x64` size inputs. We achieved state of the art results at the time. Our deep learning approach to this problem further improves the accuracy from `70.8%` to `86.2%` for the top-1 prediction accuracy and from `96.8%` to `98.4%` for top-5 prediction accuracy.

|               | Top-1 Accuracy | Top-5 Accuracy |
|---------------|:--------------:|:--------------:|
|    Leafsnap   |      70.8%     |      96.8%     |
| Deep-Leafsnap |      86.2%     |      98.4%     |

We noticed that our model failed to recognize specific classes of trees constantly causing our overall accuracy to derease. This is primarily due to the fact that those trees had very small leaves which were hard to preprocess and crop. Our training images were also resized to `64x64` due to limited computational resources. We plan on further improving our data preprocessing and increasing our image size to `224x224` in order to exceed `90%` for our top-1 prediction acurracy.

The following goes over the code and how to set it up on your own machine.

## Files
* `model.py` trains a convolutional neural network on the dataset.
* `vgg.py` PyTorch model code for VGG-16.
* `densenet.py` PyTorch model code for DenseNet-121.
* `resnet.py` PyTorch model code for ResNet.
* `dataset.py` creates a new train/test dataset by cropping the leaf and augmenting the data.
* `utils.py` helps do some of the hardcore image processing in dataset.py.
* `averagemeter.py` helper class which keeps track of a bunch of averages when training.
* `leafsnap-dataset-images.csv` is the CSV file corresponding to the dataset.
* `requirements.txt` contains the pip requirements to run the code.

## Installation
To run the models and code make sure you [Python](https://www.python.org/downloads/) installed.

Install PyTorch by following the directions [here](http://pytorch.org/).

Clone the repo onto your local machine and cd into the directory.
```
git clone https://github.com/sujithv28/Deep-Leafsnap.git
cd Deep-Leafsnap
```

Install all the python dependencies:
```
pip install -r requirements.txt
```
Make sure sklearn is updated to the latest version.
```
pip install --upgrade sklearn
```
Also make sure you have OpenCV installed either through pip or homebrew. You can check if this works by running and making sure nothing complains:
```
python
import cv2
```
Download Leafsnap's image data and extract it to the main directory by running in the directory. Original data can be found [here](http://leafsnap.com/dataset/).
```
wget https://www.dropbox.com/s/dp3sk8wpiu9yszg/data.zip?dl=0
unzip -a data.zip?dl=0
rm data.zip?dl=0
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
