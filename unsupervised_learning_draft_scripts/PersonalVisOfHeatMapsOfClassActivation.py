from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import models
from keras.applications.vgg16 import VGG16

# The local path to our target image
img_path = r'C:\Users\XGOBY\ConferenceCNN\data\sorteddata\Test\Large\Crack__20180419_13_29_14,846.bmp'.replace('\\','/')
model_path = '4th_eLu_FATIGUE_4class_model.h5'
# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(100, 100))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x)

model = load_model(model_path)


preds = model.predict(x)
print(preds)