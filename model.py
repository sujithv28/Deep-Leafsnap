import json
import numpy as np
import pandas as pd
import cv2
import os
import scipy.misc

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from PIL import Image
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Location of the training data
DATA_FILE = 'leafsnap-dataset-images.csv'
NUM_CLASSES = 185

# Load the training data into a pandas dataframe.
print('\n[INFO] Loading Dataset:')
columns = ['file_id', 'image_pat', 'segmented_path', 'species', 'source']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

images = data['image_pat']
species = data['species']

print('\n[INFO] Creating Training and Testing Data:')
images_train, images_validation, species_train, species_validation = train_test_split(
    images, species, test_size=0.25, random_state=42)

species_train = np_utils.to_categorical(species_train)
species_validation = np_utils.to_categorical(species_validation)
num_classes = species_validation.shape[1]

# Rotates an image 90 degrees
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

print('\n[INFO] Batch Generator:')
def batch_generator(images, species, batch_size=64, augment_data=True):
    # Create an array of sample indices.
    batch_images = []
    batch_species = []
    sample_count = 0
    indices = np.arange(len(images))

    while True:
        # Shuffle indices to minimize overfitting. Common procedure.
        np.random.shuffle(indices)
        for i in indices:
            path = images.iloc[i]
            # Load the center image and steering angle.
            original_image = Image.open(path.strip())
            original_image = np.array(original_image, np.float32)

            # If an image is in the incorrect orientation, rotate it
            if (original_image.shape[0] > original_image.shape[1]):
                original_image = rotate_bound(original_image, 90)

            # Resize all the images to the same size
            rotated_image = scipy.misc.imresize(rotated_image, (224,224))
            original_image = original_image / 255
            normal_species = species[i]

            batch_images.append(original_image)
            batch_species.append(normal_species)

            # Add augmentation if needed. We do this because our model is only training on plai
            # images and we want as much data as possible.
            if augment_data:
                # Flip the image left right, up down, and rotated 180
                flipped_left_right_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
                batch_images.append(flipped_left_right_image)
                batch_species.append(normal_species)

                flipped_up_down_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
                batch_images.append(flipped_up_down_image)
                batch_species.append(normal_species)

                rotated_180_image = original_image.transpose(Image.ROTATE_180)
                batch_images.append(rotated_180_image)
                batch_species.append(normal_species)

            # Increment the number of samples.
            sample_count += 1

            # If we have processed batch_size number samples or this is the last batch
            # of the epoch, then we submit the batch. Since we augment the data there is a chance
            # we have more than the number of batch_size elements in each batch.
            if (sample_count % batch_size) == 0 or (sample_count % len(images)) == 0:
                yield np.array(batch_images), np.array(batch_steering_angles)
                # Reset
                batch_images = []
                batch_species = []

activation_relu = 'relu'
learning_rate = 1e-4

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(224,224,3)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(185, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    return model

nb_epoch = 15

samples_per_epoch = 4 * len(images_train)
generator_train = batch_generator(images_train, species_train)

nb_val_samples = len(images_validation)
generator_validation = batch_generator(images_validation, species_validation, augment_data=False)

print('\n[INFO] Creating Model:')
model = VGG_16('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

print('\n[INFO] Training Model:')
model.fit_generator(generator_train,
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=nb_epoch,
                    validation_data=generator_validation,
                    nb_val_samples=nb_val_samples)

model.save_weights('model.h5', True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))

