# import all libraries

import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tqdm import tqdm
import pickle

# create model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

# add our owntop layer
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

print(model.summary())


# function for extracting feature

def feature_extrect(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# now we have to create the list of all the image name of file
import os
filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

# create feature list array and pickle it for use in other files
feature_list = [] # it is going to be 2d list

for file in tqdm(filenames):
    feature_list.append(feature_extrect(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

print("feature list array shape : {}".format(np.array(feature_list).shape))
