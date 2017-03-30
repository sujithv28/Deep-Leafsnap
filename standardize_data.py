import json
import numpy as np
import pandas as pd
import cv2
import os
import scipy.misc

import tensorflow as tf
tf.python.control_flow_ops = tf

from PIL import Image
from sklearn import preprocessing

DATA_FILE = 'leafsnap-dataset-images.csv'

print('\n[INFO] Loading Dataset:')
columns = ['file_id', 'image_pat', 'segmented_path', 'species', 'source']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

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

count = 0
for path in data['image_pat']:
	if (count == 100):
		print("[INFO] Processed %d images" % (count))
	image = Image.open(path.strip())
	image = np.array(image, np.float32)
	# If an image is in the incorrect orientation, rotate it
	if (image.shape[0] > image.shape[1]):
		rotated_image = rotate_bound(image, 90)
	else:
		rotated_image = image
	# Resize all the images to the same size
	rotated_image = scipy.misc.imresize(rotated_image, (64,64))
	cv2.imwrite(path, rotated_image)
	count = count+1
