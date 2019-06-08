# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:51:11 2018

@author: Xavier O'Rourke Goby
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import xlwt
from datetime import datetime
from numpy.random import seed
seed(1) # Fixing random seed for reproducibility


# dimensions of our images.
img_width, img_height = 100, 100

# Relative directory paths
train_data_dir = '../Data/Train'
validation_data_dir = '../Data/Validation'

# Training & Validation dataset sizes
nb_train_samples = 1600
nb_validation_samples = 800

# Arch hyperparameters
epochs = 20
batch_size = 128


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

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
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#Here I am first rescaling all my images for training. Then performing a bunch of transformations
#to my images for training <- Data Augmentation
train_datagen = ImageDataGenerator(
    # horizontal_flip=True,
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2)

#Rescaling my images to be used for voldiation. Note that I should NOT augmnet my validaiton images!
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')

history = model.fit_generator(
          # shuffle = True,
          train_generator,
          steps_per_epoch = nb_train_samples // batch_size,
          epochs = epochs,
          validation_data = validation_generator,
          validation_steps = nb_validation_samples // batch_size)


model.summary()


model.save_weights('../saved_weights/saved_weights_{0}.h5'.format(datetime.today().strftime("%Y-%m-%d")))
model.save('../saved_trained_models/trained_model_{0}.h5'.format(datetime.today().strftime("%Y-%m-%d")))

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