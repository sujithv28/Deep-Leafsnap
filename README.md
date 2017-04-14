# Deep-LeafSnap
LeafSnap replicated using deep neural networks to test accuracy compared to traditional computer vision methods. Model is built off VGG 16 for ImageNet.

## Files
* `model.py` trains a convolutional neural network on the dataset.
* `dataset.py` creates a new train/test dataset by cropping the leaf and augmenting the data.
* `utils.py` helps do some of the hardcore image processing in dataset.py.
* `Visualize-Leaf-Data.ipynb` is a jupyter notebook that explains the code.
* `leafsnap-dataset-images.csv` is the CSV file corresponding to the dataset.
* `requirements.txt` contains the pip requirements to run the code.

## Installation
To run the models and code make sure you [Python](https://www.python.org/downloads/) installed.

Install Tensorflow following the instructions [here](https://www.tensorflow.org/install/).

Install PyTorch by following the directions [here](http://pytorch.org/).

Clone the repo onto your local machine and cd into the directory.
```
git clone https://github.com/sujithv28/Deep-LeafSnap.git
cd Deep-LeafSnap
```

Install all the python dependencies:
```
pip install -r requirements.txt
```
Make sure Keras and sklearn are updated to the latest version.
```
pip install --upgrade keras
pip install --upgrade sklearn
```
Set Keras to use Tensorflow as backend:
```
python
import keras
quit()
nano ~/.keras/keras.json
```
Change your values to look something like:
```
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
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
This cleans the dataset by cropping only neccesary portions of the images containing the leaves and also resizes them to `64x64`.

## Training Model
To train the model, run
```
python model.py
```
