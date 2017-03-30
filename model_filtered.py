import json
import numpy as np
import pandas as pd
import cv2
import os
import scipy.misc

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Lambda
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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

lab_data = data.drop(data[data['image_pat'].str.contains("field")].index)
field_data = data.drop(data[data['image_pat'].str.contains("lab")].index)
# lab_data = [~data['image_pat'].str.contains("field")]
# field_data = [~data['image_pat'].str.contains("lab")]

images = lab_data['image_pat']
species = lab_data['species']

print('\n[INFO] Creating Training and Testing Data:')
images_train, images_validation, species_train, species_validation = train_test_split(
    images, species, test_size=0.15, random_state=42)

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
            rotated_image = scipy.misc.imresize(rotated_image, (64,64))
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

def create_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 62, 62), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation=activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation=activation_relu))
    model.add(Dense(50, activation=activation_relu))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

nb_epoch = 20

samples_per_epoch = 4 * len(images_train)
generator_train = batch_generator(images_train, species_train)

nb_val_samples = len(images_validation)
generator_validation = batch_generator(images_validation, species_validation, augment_data=False)

print('\n[INFO] Creating Model:')
model = create_model()
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

