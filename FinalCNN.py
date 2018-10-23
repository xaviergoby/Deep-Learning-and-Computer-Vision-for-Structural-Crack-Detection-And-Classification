# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:51:11 2018

@author: Xavier O'Rourke Goby
"""

#Fixing random seed for reproducibility
from numpy.random import seed
seed(1)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils.generic_utils import get_custom_objects
import xlwt

#############Swish activation function deifinition##########
#
#class Swish(Activation):
#    
#    def __init__(self, activation, **kwargs):
#        super(Swish, self).__init__(activation, **kwargs)
#        self.__name__ = 'swish'
#
#
#def swish(x):
#    return K.sigmoid(x) * x
#
#get_custom_objects().update({'swish': Swish(swish)})
#
## To use the Swish activation function, replace Activation('ActivationFunctionName') with Swish('swish')


# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = 'TextImageClassificationData/Train'
validation_data_dir = 'TextImageClassificationData/Validation'
nb_train_samples = 499
nb_validation_samples = 97
epochs = 20
batch_size = 10
#val 800
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



model = Sequential()
###############Image "Layer"##############
#The shape of my input image is (100,100,3)
#Where img_width = 100 pixels, img_height = 100 pixels, # of channels  = 3 (RGB) 
#The shape of my input batch is (bs,100,100,3) where bs = # of images/samples per batch/batch size
##########################################


#############Convolutional 2D Layer#################
#My 1st conv layer takes an input feature map of (100,100,3) 
#My conv layer then performs a convolution. This invovles a filter(W) slidding/scanning over the input feature map
#basically my image, with a size of (3,3,3) where the last digit 3 is the # of "planes" in the previous layer AKA the 
#the dimension of the depth axis for a coloured image AKA channels axis
#The patch over which a filter is currently scanning/filtering over on the feature map/image is called the receptive
#field and it has the same size as the fitler!

#The process of my filter passing over a receptive field will result in 3*3*3= 27 weight paramenters and 1 bias parameter
#to give a total of 28 parameters for each fitler/neuron/kernel.

#32 in my Conv2D layer specifies the number of filters/neurons/kernels in my layer. This therefore means that my
#1st Conv2D layer will have a total of 28*32 = 896 (trainable) parameters.
####################################################

model.add(Conv2D(16, (3, 3), input_shape=input_shape)) 

#######Element-Wise Threshold############

model.add(Activation('relu'))

############ Pooling Layer  ##############

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#Constructing my densely connected classifier network: a stack of Dense layers.
#These classifiers process vectors which are 1D whereas the curren output of my previous layer
#is a 3D tensor of shape (3,3,64)


#First I must flatten my 3D outputs of shape (3,3,64) into vecctors of shape (576, ) before they are fed 
#through my two Dense layers.
#
model.add(Flatten()) 
model.add(Dense(16))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#Here I am first rescaling all my images for training. Then performing a bunch of transformations
#to my images for training <- Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#Rescaling my images to be used for voldiation. Note that I should NOT augmnet my validaiton images!
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history=model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)


model.summary()


model.save_weights('images_with_and_Without_text.h5')
model.save('images_with_and_Without_text.h5')

################Plotting Performance: Accuracy and Loss of Trainning and Validation###############

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'], 'bo')
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], 'bo')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

############Create Excel sheet with config data and acc and loss for training and validation to compare configs########
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")
sheet1.write(0, 0, "Test data - [X]") #In [x] fill the aim of this test - a different activation function, added layers, different layer size, etc.
sheet1.write(1, 0, "Batch size")
sheet1.write(1, 1, batch_size)
sheet1.write(2, 0, "Epochs")
sheet1.write(2, 1, epochs)
sheet1.write(3, 0, "Conv2D size")
sheet1.write(3, 1, "32, 32, 64") #Sadly needs to be written manually every time
sheet1.write(4, 0, "Deep layer size")
sheet1.write(4, 1, "16, 16, 16, 16, 4") #Again, needs to be written manually again
sheet1.write(5, 0, "Training Accuracy")
sheet1.write(5, 1, "Validation Accuracy")
sheet1.write(5, 2, "Training Loss")
sheet1.write(5, 3, "Validation Loss")
#appends all data in columns. If you can do this better without wasting too much time on other more important stuff by all means do it
i = 5
j = 5
k = 5
l = 5
for n in history.history['acc']:
    i += 1
    sheet1.write(i, 0, n)
    
for n in history.history['val_acc']:
    j += 1
    sheet1.write(j, 1, n)
    
for n in history.history['loss']:
    k += 1
    sheet1.write(k, 2, n)
    
for n in history.history['val_loss']:
    l += 1
    sheet1.write(l, 3, n)
    
book.save("setting4Cls1[X]xls") #Do not run with the same book.save name twice, it will replace the first sheet without prompt