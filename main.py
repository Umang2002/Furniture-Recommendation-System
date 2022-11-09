# import all the libraries
import streamlit
import os
import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPool2D
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm

# nearest neighbores algoritham
from sklearn.neighbors import NearestNeighbors
import cv2


## FOR IMAGE
import sys
from PIL import Image
sys.modules['Image'] = Image

from PIL import Image
print(Image.__file__)

import Image


# create model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

# add our owntop layer
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

# extract the pickle feature list and file names
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


streamlit.title("Furniture Recommendation System")


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,fetaure_list):
    # apply the nearest neighbors algoritham to find the closest five recommendations
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    # we get the distances and indics of the recommeded product
    distances, indices = neighbors.kneighbors([features])

    return indices
# steps
# file upload
uploaded_file = streamlit.file_uploader("Choose an image")





if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # file has been uploaded
        # display the file
        display_img = Image.open(uploaded_file)
        streamlit.image(display_img)

        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)

        # recommendation

        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5 = streamlit.columns(5)

        with col1:
            streamlit.image(filenames[indices[0][0]])
        with col2:
            streamlit.image(filenames[indices[0][1]])
        with col3:
            streamlit.image(filenames[indices[0][2]])
        with col4:
            streamlit.image(filenames[indices[0][3]])
        with col5:
            streamlit.image(filenames[indices[0][4]])

    else:
        streamlit.header("Error in uploading file")





