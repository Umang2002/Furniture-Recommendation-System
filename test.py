# import all the libraries
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

# now perform all the steps with single image and new image to perform recommendation
img = image.load_img('sample/table_sample.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array,axis=0)
preprocessed_img= preprocess_input(expanded_img_array)

result = model.predict(preprocessed_img).flatten()
normalized_result = result/norm(result)

# apply the nearest neighbors algoritham to find the closest five recommendations
neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

# we get the distances and indics of the recommeded product
distances,indices= neighbors.kneighbors([normalized_result])

print("indics of recommendead product is : {}".format(indices))
print("distances of recommendead product is : {}".format(distances))

for file in indices[0][0:5]:
    print(filenames[file])

# use opencv to view the recommended product
for file in indices[0][0:5]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',temp_img)
    cv2.waitKey(0)



