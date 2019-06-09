# -*- coding: utf-8 -*-
"""
Created on 09/10/2018

@author: Xavier O'Rourke Goby
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

# img_path = r"C:\Users\XGOBY\Pictures\compresseddata\no_crack_compressed\Crack__20180418_20_22_14,878.bmp".replace('\\','/')

img_path = 'data/sorteddata/Test/Small/Crack__20180419_00_26_30,659.bmp'
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

img = cv2.imread(img_path)

textboxstr = "Large Crack:{0}\nMedium Crack:{1}\nSmall Crack:{2}\nNo Crack:{3}".format(large_crack_prob_str, medium_crack_prob_str, small_crack_prob_str, no_crack_prob_str)

fig = plt.figure()
# fig.suptitle('Image Being Predicted\n',fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.5)
ax.set_title("""\nTrue Crack Classification Label: {0}
\nPredicted Crack Classification Label: {1}""".format(true_class_label(img_path),pred_class_label(class_probs)[1]))

ax.text(0.95, 0.01, textboxstr,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
# ax.plot([2], [1], 'o')
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()