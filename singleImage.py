from tensorflow.keras.preprocessing import image
import numpy as np
from numpy.linalg import norm
from PIL import Image

# transfer learning module import
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D

# load image
img = image.load_img('images/bed_0.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
print("Image array size: {}".format(img_array.shape))

# before giving to model it require particular size and in the batches so convert in the batch of 1
expanded_img_array = np.expand_dims(img_array,axis=0)
print("Expanded arary batch size : {}".format(expanded_img_array.shape))

# train model on our own and create model on top of it
model = ResNet50(weights='imagenet',include_top =False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# it rrquire specific format of the data as input
# 1) convert in particular range meaning normalize the input
# 2) convert the format of imgae form RGB TO BGR
preprocesed_img = preprocess_input(expanded_img_array)

# predict the output
# model.predict(preprocesed_img)
# model.predict(preprocesed_img).shape => (1,2048)
# model.predict(preprocesed_img).flatten().shape => (2048,)
# L2 norm meaning => np.sqrt(np.dot(model.predict(preprocesed_img).flatten(),model.predict(preprocesed_img).flatten()))
# L2 norm each number squared sum

print(model.predict(preprocesed_img)/norm(model.predict(preprocesed_img).flatten()))


