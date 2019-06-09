import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import backend as K
import cv2



classlabels = ['Large','Medium','None','Small']
img_path = 'data/sorteddata/Test/None/Crack__20180418_15_56_27,679.bmp'
img_name = img_path.split('/')[-1]

def img_class_label(img_path):
    img_path_split = img_path.split('/')
    for i in classlabels:
        if i in img_path_split:
            return i

model_path = '4th_eLu_FATIGUE_4class_model.h5'

model = load_model(model_path)
print(model.summary())

img = image.load_img(img_path, target_size=(100, 100))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)




layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
print(activations)
print(activations[26])

first_layer_activation = activations[0]
print(first_layer_activation.shape)

preds = model.predict(img_tensor)
print(preds)

# idx = np.argmax(preds[0])
pred_label_idx = np.max(preds[0])

plt.imshow(img_tensor[0])
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
# plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()

idx = np.argmax(preds[0])

large_crack_output = model.output[:, idx]
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])




def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


kept_filters = []
for filter_index in range(64):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)


    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict['activation_77'].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, img_tensor)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([img_tensor], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, 100, 100))
    else:
        input_img_data = np.random.random((1, 100, 100, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break


    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x


    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    print('Filter %d processed in %ds')