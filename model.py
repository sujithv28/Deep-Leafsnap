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

# GLOBAL CONSTANTS
DATA_FILE = 'leafsnap-dataset-images.csv'
NUM_CLASSES = 185
NB_EPOCH = 15
VGG_WEIGHTS_FILE = 'vgg16_weights.h5'

print('\n[INFO] Loading Dataset:')
columns = ['file_id', 'image_pat', 'segmented_path', 'species', 'source']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

print('\n[INFO] Creating Training and Testing Data (75-25 Split):')
images_train, images_validation, species_train, species_validation = train_test_split(
    data['image_pat'], data['species'], test_size=0.25, random_state=42)

species_train = np_utils.to_categorical(species_train)
species_validation = np_utils.to_categorical(species_validation)
num_classes = species_validation.shape[1]

# Rotates an image 90 degrees
def check_and_rotate(image, angle):
    if (image.shape[0] > image.shape[1]):
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
    else:
        return image

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
            original_image = Image.open(path.strip())
            original_image = np.array(original_image, np.float32)
            # original_image = check_and_rotate(original_image, 90)

            # Resize all the images to the same size
            original_image = scipy.misc.imresize(original_image, (224,224))
            original_species = species[i]

            batch_images.append(original_image)
            batch_species.append(original_species)

            # Add augmentation if needed. We do this because our model is only training on plai
            # images and we want as much data as possible.
            if augment_data:
                # Flip the image left right, up down, and rotated 180 to augment data
                flipped_left_right_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
                batch_images.append(flipped_left_right_image)
                batch_species.append(original_species)

                flipped_up_down_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
                batch_images.append(flipped_up_down_image)
                batch_species.append(original_species)

                rotated_180_image = original_image.transpose(Image.ROTATE_180)
                batch_images.append(rotated_180_image)
                batch_species.append(original_species)

            # Increment the number of samples.
            sample_count += 1

            # If we have processed batch_size number samples or this is the last batch
            # of the epoch, then we submit the batch. Since we augment the data there is a chance
            # we have more than the number of batch_size elements in each batch.
            if (sample_count % batch_size) == 0 or (sample_count % len(images)) == 0:
                yield np.array(batch_images), np.array(batch_species)
                # Reset
                batch_images = []
                batch_species = []

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
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

print('\n[INFO] Generating training and validation batches')
samples_per_epoch = 4 * len(images_train)
generator_train = batch_generator(images_train, species_train, augment_data=True)
nb_val_samples = len(images_validation)
X_validation,y_validation = batch_generator(images_validation, species_validation, augment_data=False)

print('\n[INFO] Creating Model:')
model = VGG_16(VGG_WEIGHTS_FILE)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

print('\n[INFO] Training Model:')
model.fit_generator(generator_train,
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=NB_EPOCH,
                    validation_data=(X_validation,y_validation),
                    nb_val_samples=nb_val_samples)

model.save_weights('model.h5', True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

score = model.evaluate(X_validation, y_validation, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
