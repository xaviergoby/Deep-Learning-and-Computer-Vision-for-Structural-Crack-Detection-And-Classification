# -*- coding: utf-8 -*-
"""
Created on 09/10/2018

@author: Xavier
"""
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text as text
import matplotlib.image as mpimg
#1st_FATIGUE_4class_model.h5
from keras import models

# img_path = r"C:\Users\XGOBY\Pictures\compresseddata\no_crack_compressed\Crack__20180418_20_22_14,878.bmp".replace('\\','/')

img_path = 'data/sorteddata/Test/Large/Crack__20180419_13_33_41,453.bmp'
model_path = '4th_eLu_FATIGUE_4class_model.h5'

model = load_model(model_path)
classlabelmsgs = ['A large crack is present in the image','A medium crack is present in the image','No crack is present in the image','A small crack is present in the image']
classlabels = ['Large','Medium','None','Small']


def img_transformer(img_path):
    img_width, img_height = 100, 100
    img = image.load_img((img_path), target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    myimage = np.vstack([x])
    return myimage

class_probs = model.predict(img_transformer(img_path))
class_probs *= 100

def pred_class_label(class_probs):
    if class_probs.max() == class_probs[0][0]:
        # print(classlabelmsgs[0])
        return classlabelmsgs[0],classlabels[0]
    elif class_probs.max() == class_probs[0][1]:
        # print(classlabelmsgs[1])
        return classlabelmsgs[1],classlabels[1]
    elif class_probs.max() == class_probs[0][2]:
        # print(classlabelmsgs[2])
        return classlabelmsgs[2],classlabels[2]
    else:
        # print(classlabelmsgs[3])
        return classlabelmsgs[3],classlabels[3]


def true_class_label(img_path):
    return img_path.split('/')[3]


classes = model.predict_classes(img_transformer(img_path))
print("\nArray of prediction probabilities:",class_probs)
print('\nLargest prediction probabilities array element message: This image belongs to bin/element number', classes, 'in the prob class array above')
large_crack_prob_str = str(np.around(class_probs[0][0], decimals = 1)) +'%'
medium_crack_prob_str = str(np.around(class_probs[0][1], decimals = 1)) +'%'
small_crack_prob_str = str(np.around(class_probs[0][3], decimals = 1)) +'%'
no_crack_prob_str = str(np.around(class_probs[0][2], decimals = 1)) +'%'

prediction_msg = pred_class_label(class_probs)

print("\nPrediction message:", prediction_msg[0])
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print('The percentage probabilities computed for each possible label/class are:\n')
print('A large crack is present in the image:', large_crack_prob_str)
print('A medium crack is present in the image:', medium_crack_prob_str)
print('No crack is present in the image:', no_crack_prob_str)
print('A small crack is present in the image:', small_crack_prob_str)

layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_transformer(img_path))
# last_layer_activation = activations[26]
# eigth_layer_activation = activations[7]
# first_layer_activation = activations[0]
# print("Last layer activations:",last_layer_activation, last_layer_activation.shape)
# print("First layer activations:",first_layer_activation, first_layer_activation.shape)

# plt.matshow(eigth_layer_activation[0, :, :, 63], cmap='viridis')
# plt.show()

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers:
    layer_names.append(layer.name)

images_per_row = 4


for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

        # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
