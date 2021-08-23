import os
from pathlib import Path
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from vgg_model import VGG16
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


class VGG16_model():
    def __init__(self,image_dataset_path,dataset_path):
        self.image_path = image_dataset_path
        self.dataset = dataset_path
    

    def preprocess_image(image_name):
        '''function: to convert input images into numpy array and preprocess it

                parameter: 
                image file path

                Returns:
                image numpy array 
        '''
        directory = Path(self.image_path)
        abs_file_path = directory / image_name
        img = load_img(abs_file_path, target_size=(224, 224))
        image_array = img_to_array(img)
        img_expand = np.expand_dims(image_array, axis=0)
        preprocessed_img = preprocess_input(img_expand)
        return preprocessed_img

    
    def feature_extractor(images: list):
        '''
        function: to extract features using pretrained VGG16 model

          parameter:
          image list: list of all images present in image path

          returns:
          a list of  embedded vectors
        '''
        features=[]
        #load a previously-trained neural network, that of VGG16, which comes with Keras.
        model = VGG16(weights='imagenet', include_top=True)
        # copy the model, but remove the last layer (the classification layer),fc2 dense (1,1,4096)
        feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
        # use VGG to extract features
        for image_name in images:
            x = preprocess_image(image_name)
            vgg_feat = feat_extractor.predict(x)[0]
            features.append(vgg_feat)
        return np.array(features)

    def pearson_correlation(self, vector1, vector2):
        '''
        function: to find the correlation between images

          parameter:
          vector1: image embedded vector
          vector2: image embedded vector

          returns:
          correlation score between 
        '''
        similarity_score = cosine_similarity(vector1, vector2)
        df_sim = np.float32(self.dataset['sim']).transpose()
        score = stats.pearsonr(df_sim, similarity_score)
        return score


