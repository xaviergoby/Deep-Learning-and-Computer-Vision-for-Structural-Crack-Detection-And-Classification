from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import layers

img_path = 'data/sorteddata/Test/Large/Crack__20180419_13_33_41,453.bmp'
model_path = '4th_eLu_FATIGUE_4class_model.h5'
model = load_model(model_path)
model_weights = model.get_weights()
model_layers = model.layers

# config = model_layers[0].get_config()
# layer = layers.deserialize({'class_name': model_layers[0].__class__.__name__,
#                             'config': config})


def img_transformer(img_path):
    img_width, img_height = 100, 100
    img = image.load_img((img_path), target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    myimage = np.vstack([x])
    return myimage

x = img_transformer(img_path)
preds = model.predict(x)
x = preprocess_input(x)

