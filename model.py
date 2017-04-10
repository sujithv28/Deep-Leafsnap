import json
import numpy as np
import pandas as pd
import cv2
import os
import scipy.misc
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats, integrate

# GLOBAL CONSTANTS
DATA_FILE = 'leafsnap-dataset-images.csv'
NUM_CLASSES = 185
NB_EPOCH = 15
VGG_WEIGHTS_FILE = 'vgg16_weights.h5'

print('\n[INFO] Loading Dataset:')
columns = ['file_id', 'image_pat', 'segmented_path', 'species', 'source']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

print('\n[INFO] Creating Training, Validation, and Testing Data (75-15-10 Split):')
X_train, images_rest, y_train, species_rest = train_test_split(
    data['image_pat'], data['species'], test_size=0.25, random_state=42)
# Divide the remaining 25% into validation and testing (60-40)
X_validation, X_test, y_validation, y_test = train_test_split(
    images_rest, species_rest, test_size=0.4, random_state=42)

encoder = preprocessing.LabelEncoder()
encoder.fit(y_train)
classes = encoder.classes_

y_train = encoder.transform(y_train)
y_train = np_utils.to_categorical(y_train)

y_validation = encoder.transform(y_validation)
y_validation = np_utils.to_categorical(y_validation)

y_test = encoder.transform(y_test)
y_test = np_utils.to_categorical(y_test)

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

def load_image_and_preprocess(path, flip_left_right=False, flip_up_down=False, rotate_180=False):
    # Open image from disk and flip it if generating data.
    image = Image.open(path.strip())

    if flip_left_right:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_up_down:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_180:
        image = image.transpose(Image.ROTATE_180)

    # Convert the image into mulitdimensional matrix of float values (normally int which messes up our division).
    image = np.array(image, np.float32)
    # Resize Image
    image = scipy.misc.imresize(image, (224,224))
    # Return the modified image.
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
            original_species = species[i]

            image = load_image_and_preprocess(path)
            batch_images.append(image)
            batch_species.append(original_species)pipp

            # Add augmentation if needed. We do this because our model is only training on plai
            # images and we want as much data as possible.
            if augment_data:
                # Flip the image left right, up down, and rotated 180 to augment data
                flipped_left_right_image = load_image_and_preprocess(path,flip_left_right=True)
                batch_images.append(flipped_left_right_image)
                batch_species.append(original_species)

                flipped_up_down_image = load_image_and_preprocess(path,flip_up_down=True)
                batch_images.append(flipped_up_down_image)
                batch_species.append(original_species)

                rotated_180_image = load_image_and_preprocess(path,rotate_180=True)
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

def VGG_16():
    base_model = VGG16(weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    input = Input(shape=(224,224,3),name = 'image_input')
    output_vgg16_conv = base_model(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(input=input, output=x)
    return model

print('\n[INFO] Generating training and validation batches')
steps_per_epoch = 4 * len(X_train)
generator_train = batch_generator(X_train, y_train, augment_data=True)
validation_steps = len(X_validation)
generator_validation = batch_generator(X_validation, y_validation, augment_data=False)

print('\n[INFO] Creating Model:')
model = VGG_16()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
print(model.summary())

best_model_file = "vgg_model.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

print('\n[INFO] Training Model:')
history = model.fit_generator(generator_train,
                              steps_per_epoch=steps_per_epoch,
                              nb_epoch=NB_EPOCH,
                              validation_data=generator_validation,
                              validation_steps=validation_steps,
                              verbose=0,
                              callbacks=[best_model])

print('[INFO] Saving the best model...')
with open('vgg_model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

print('[INFO] Loading the best model...')
model = load_model(best_model_file)

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('[INFO] Prediction:')
print(model.predict_classes(X_test))
