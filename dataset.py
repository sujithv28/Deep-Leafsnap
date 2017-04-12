import json
import numpy as np
import pandas as pd
import cv2
import os
import scipy.misc
import random
import time

from PIL import Image
from sklearn import preprocessing
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import stats, integrate, misc
from skimage import io,img_as_float

# GLOBAL CONSTANTS
DATA_FILE = 'leafsnap-dataset-images.csv'
NUM_CLASSES = 185

columns = ['file_id', 'image_pat', 'segmented_path', 'species', 'source']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

images     = data['image_pat']
images_seg = data['segmented_path']
species    = data['species']
species_classes = sorted(set(species))
print('Number of Samples: {}'.format(len(images)))

def load_image_and_preprocess(path, segmented_path, flip_left_right=False, flip_up_down=False, rotate_180=False):
    # Open image from disk and flip it if generating data.
    image = misc.imread(path.strip())
    segmented_image = misc.imread(segmented_path.strip())

    img = segmented_image
    h, w = img.shape[:2]
    height,width = h,w
    # print('Height: {:3d}, Width: {:4d}\n'.format(height,width))
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate bounding rectangles for each contour.
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    if rects == []:
        # No segmentation in image
        return []
    else:
        #Calculate the combined bounding rectangle points.
        top_y = max(0, min([y for (x, y, w, h) in rects]) - 40)
        bottom_y = min(height, max([y+h for (x, y, w, h) in rects]) + 80)
        left_x = max(0, min([x for (x, y, w, h) in rects]) - 40)
        right_x = min(width-1, max([x+w for (x, y, w, h) in rects]) + 80)
        if left_x >= 455 or top_y >= 480:
            return []

    # Use the rectangle to crop on original image
    img = image[top_y:bottom_y, left_x:right_x]
    img = scipy.misc.imresize(img, (224,224))
    return img

print('[INFO] Processing Images')
cropped_images = []
image_species = []
image_paths = []
count = 0
cwd = os.getcwd()
os.mkdir('images')
for index in range(len(images)):
    image = load_image_and_preprocess(images[index], images_seg[index])
    if type(image) != type([]):
        file_name = '{}.jpg'.format(count)
        print(file_name)

        image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dirname, file_name), image_to_write)

        image_paths.append('images/' + file_name)
        cropped_images.append(image)
        image_species.append(species[index])

        count += 1
    if index > 0 and index%1000 == 0:
        print('[INFO] Processed {:5d} images'.format(index))

print('[DONE]')