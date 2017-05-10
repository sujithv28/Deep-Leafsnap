import cv2
import json
import numpy as np
import os
import pandas as pd
import torch
import random

from PIL import Image
from scipy import integrate
from scipy import misc
from scipy import stats
from skimage import img_as_float
from skimage import io
from sklearn import preprocessing


def rotate(image, angle):
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


def load_image_and_preprocess(path, segmented_path):
    # Open image from disk
    image = misc.imread(path.strip())
    segmented_image = misc.imread(segmented_path.strip())

    img = segmented_image
    h, w = img.shape[:2]
    height, width = h, w
    # print('Height: {:3d}, Width: {:4d}\n'.format(height,width))
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate bounding rectangles for each contour.
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    if rects == []:
        # No contours in image (pure black) just keep entire image
        top_y = 0
        bottom_y = height - 120
        left_x = 0
        right_x = width0 = -120
    else:
        # Calculate the combined bounding rectangle points.
        top_y = max(0, min([y for (x, y, w, h) in rects]) - 40)
        bottom_y = min(height, max([y + h for (x, y, w, h) in rects]) + 80)
        left_x = max(0, min([x for (x, y, w, h) in rects]) - 40)
        right_x = min(width - 1, max([x + w for (x, y, w, h) in rects]) + 80)

        # Prevent Tile Out of Bounds Issues
        if top_y == bottom_y:
            bottom_y = min(height, bottom_y + 1)
            top_y = max(0, top_y - 1)
        if left_x == right_x:
            right_x = min(width, right_x + 1)
            left_x = max(0, left_x - 1)

        # Landscape Image
        if width >= height:
            if left_x >= 450:
                right_x = width - 150
                left_x = 0
            if top_y >= 350 or bottom_y <= 200:
                top_y = 0
                bottom_y = height - 100
        # Portrait
        else:
            if left_x >= 350:
                right_x = width - 100
                left_x = 0
            if top_y >= 450 or bottom_y <= 250:
                top_y = 0
                bottom_y = height - 150

    # Use the rectangle to crop on original image
    img = image[top_y:bottom_y, left_x:right_x]
    img = misc.imresize(img, (224, 224))
    return img


def paths_to_images(image_paths, species, augment_data=False):
    batch_images = []
    batch_species = []
    indices = range(len(image_paths))
    np.random.shuffle(indices)

    count = 0
    for i in indices:
        image = load_image_and_preprocess(image_paths[i], 0)
        batch_images.append(image)
        batch_species.append(species[i])

        if augment_data:
            angles = [45, 90, 135, 180, 225, 270, 315]
            random.shuffle(angles)
            for i in range(4):
                rotated_image = load_image_and_preprocess(image_paths[i], angles[i])
                batch_images.append(rotated_image)
                batch_species.append(species[i])

        if count > 0 and count % 1000 == 0:
            print('[INFO] Processed {:5d} paths'.format(count))
        count += 1

    return np.array(batch_images), np.array(batch_species)
