# -*- coding: utf-8 -*-
"""
Created on 09/10/2018

@author: Xavier O'Rourke Goby
"""
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
# Data/Test/Large/Crack__20180419_13_29_14,846.bmp
# Data/Test/Medium/Crack__20180419_06_19_59,025.bmp
# saved_trained_models/TrainedModel_elu.h5
# img_path = '../Data/Test/Large/Crack__20180419_13_29_14,846.bmp' # Relative path of an image in the Test folder which you would like to test
# img_path = '../Data/Test/Medium/Crack__20180419_06_19_59,025.bmp' # Relative path of an image in the Test folder which you would like to test
img_path = "../Data/Test/Medium/Crack__20180419_06_16_35,563.bmp"
# img_path = "../Data/Test/Medium/Crack__20180419_06_19_09,915.bmp"
model_path = '../saved_trained_models/TrainedModel_elu.h5' # The path of the saved (trained) model
classlabelmsgs = ['A large crack is present in the image','A medium crack is present in the image',
                  'No crack is present in the image','A small crack is present in the image']
classlabels = ['Large','Medium','None','Small']
model = load_model(model_path)


def img_transformer(img_path):
    """
    This function is meant for performing the operation of loading the image then preprocessing it in the same manner as it was
    done for each and every one of the images which were supplied to the FinalCNN.py script for training
    :param img_path: The relative path of an image you wish to test out
    :return: the preprocessed image
    """
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
    """
    This function is simply meant for facilitating the job of extracting the bin with the highest value
    within the probability class array
    :param class_probs: probability class array
    :return: The class label and class label message corresponding with the bin (element) in the probability class array
    with the highest value
    """
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
    """
    This function is simply meant for extracting the true class label of the test image
    :param img_path: The relative path of an image you wish to test out
    :return: The true class label of the image
    """
    for i in img_path.split('/'):
        if i in classlabels:
            return i

classes = model.predict_classes(img_transformer(img_path))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
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
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

img = cv2.imread(img_path)
textboxstr = "Large Crack:{0}\nMedium Crack:{1}\nSmall Crack:{2}\nNo Crack:{3}".format(large_crack_prob_str,
                                                                                       medium_crack_prob_str,
                                                                                       small_crack_prob_str,
                                                                                       no_crack_prob_str)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.5)
ax.set_title("""\nTrue Crack Classification Label: {0}
\nPredicted Crack Classification Label: {1}""".format(true_class_label(img_path),pred_class_label(class_probs)[1]))
ax.text(0.95, 0.01, textboxstr,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
