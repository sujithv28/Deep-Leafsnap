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

images = data['image_pat']
images_seg = data['segmented_path']
species = data['species']
species_classes = sorted(set(species))
print('[INFO] Number of Samples: {}'.format(len(images)))

def load_image_and_preprocess(path, segmented_path):
    # Open image from disk
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
        # No contours in image (pure black) just keep entire image
        top_y = 0
        bottom_y = height-120
        left_x = 0
        right_x = width0=-120
    else:
        #Calculate the combined bounding rectangle points.
        top_y = max(0, min([y for (x, y, w, h) in rects]) - 40)
        bottom_y = min(height, max([y+h for (x, y, w, h) in rects]) + 80)
        left_x = max(0, min([x for (x, y, w, h) in rects]) - 40)
        right_x = min(width-1, max([x+w for (x, y, w, h) in rects]) + 80)

        # Prevent Tile Out of Bounds Issues
        if top_y == bottom_y:
        	bottom_y = min(height, bottom_y+1)
        	top_y = max(0, top_y-1)
        if left_x == right_x:
        	right_x = min(width, right_x+1)
        	left_x = max(0, left_x-1)

        # Landscape Image
        if width >= height:
        	if left_x >= 450:
        		right_x = width-150
        		left_x = 0
        	if top_y >= 350 or bottom_y <= 200:
        		top_y = 0
        		bottom_y = height-100
        # Portrait
        else:
        	if left_x >= 350:
        		right_x = width-100
        		left_x = 0
        	if top_y >= 450 or bottom_y <= 250:
        		top_y = 0
        		bottom_y = height-150

    # Use the rectangle to crop on original image
    img = image[top_y:bottom_y, left_x:right_x]
    img = scipy.misc.imresize(img, (224,224))
    return img

cropped_images = []
image_species = []
image_paths = []
if not os.path.exists('dataset/cropped'):
	os.mkdir('dataset/cropped')

print('[INFO] Processing Images')
for index in range(len(images)):
    image = load_image_and_preprocess(images[index], images_seg[index])
    if type(image) != type([]):
    	image_dir = 'dataset/cropped/{}'.format(species[index])
    	if not os.path.exists(image_dir):
    		os.mkdir(image_dir)

        file_name = '{}.jpg'.format(index)

        image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(image_dir, file_name), image_to_write)
        image_paths.append(os.path.join(image_dir, file_name))
        cropped_images.append(image)
        image_species.append(species[index])

    if index > 0 and index%1000==0:
        print('[INFO] Created {:5d} images'.format(index))

print('[INFO] Final Number of Samples: {}'.format(len(image_paths)))

print('[INFO] Saving CSV File')
indices = range(len(image_paths))
indices = [x+1 for x in indices]

df = pd.DataFrame({'image_paths': image_paths,
				   'species'    : image_species},
				   index=indices)
df.to_csv('leafsnap-dataset-cropped-images.csv', sep='\t')

new_species = sorted(set(image_species))
print('[INFO] Species: {}'.format(new_species))
print('[INFO] Number of species: {}'.format(len(new_species)))

print('[DONE]')
