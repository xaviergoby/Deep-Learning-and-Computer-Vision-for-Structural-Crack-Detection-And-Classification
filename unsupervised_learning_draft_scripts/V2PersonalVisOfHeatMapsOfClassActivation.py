# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:40:51 2018

@author: Xavier
"""
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.text as text
import matplotlib.image as mpimg
from keras.applications.vgg16 import preprocess_input, decode_predictions


img_path = r'C:\Users\XGOBY\ConferenceCNN\data\sorteddata\Test\Large\Crack__20180419_13_29_14,846.bmp'.replace('\\','/')
model_path = '4th_eLu_FATIGUE_4class_model.h5'

im = Image.open(img_path)
width, height = im.size
print(im.size)

# Dimensions of my images
img_width, img_height = 100, 100

# Load trained model
model = load_model(model_path)

# Loading my images with the specific dimensions
img = image.load_img((img_path), target_size=(img_width, img_height))

# Converting my input image "img" to a Numpy array "x" with a shape of (3,150,150)
x = image.img_to_array(img)

print(x)
x = x / 255
print(x)

x = np.expand_dims(x, axis=0)  # This function expands the array by inserting a new axis
# at the specified position. Two parameters are required by this function.
print(x)

images = np.vstack([x])

print(images)

class_probs = model.predict(images)
class_probs *= 100
print('The probability class array is:', np.around(class_probs, decimals=1), '\n')

from keras import layers, models

all_layers = []
input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
prev_layer = input_layer
for layer in model.layers:
    all_layers.append(layer)
    prev_layer = layer(prev_layer)

print("all my layers:",all_layers)
print(type(all_layers))

funcmodel = models.Model([input_layer], [prev_layer])
