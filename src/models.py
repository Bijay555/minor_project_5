import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.stats import pearsonr
from scipy.spatial import distance


class MobilenetEmbedding:
    """
    A class used to extract feature vector of images using Mobilenet model

    ...

    Attributes
    ----------
    module : 
        Module to hold the loaded model from url

    img_dir_path : str
        Path to the image directory

    Methods
    -------
    preprocess_image(img_path) 
        Preprocess images to make it comptabile as an input to the Mobilenet model

    get_feature_vector(img_df, column_name)
        Computes and returns feature vectors for images in dataframe column

    similarity_score(img1_embeddings, img2_embeddings)
        Computes and returns list of cosine similiarity score

    pearson_coeff(sim_scores, other_sim_scores)
        Computes and returns pearson correlation coefficient
    """

    def __init__(self, image_dir_path):
        """
        Creates count vectorizer and fits the dataframe column.

        Parameters
        ----------
        image_dir_path(str) : Path to the image directory
        """

        self.module = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5")
        self.image_dir_path = image_dir_path

    def preprocess_image(self, img_path):
        """
        Preprocess the column of dataframe. 

        Parameters
        ----------
        img_path(str) : image path
        
        Returns
        -------
        img(tensorflow.python.framework.ops.EagerTensor) : processed image 
        """

        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        return img

    def get_feature_vector(self, img_df, column_name):
        """
        Computes and returns feature vectors for images in dataframe column

        Parameters
        ----------
        img_df(pandas.core.frame.DataFrame) : pandas dataframe
        column_name(str) : column name

        Returns
        -------
        embedding_vectors(list) : list of feature vectors
        """
        embedded_vectors = []
        for idx, value in img_df[column_name].iteritems():
            file_name = self.image_dir_path+"/"+str(value)
            img = self.preprocess_image(file_name)
            feature_vector = self.module(img)
            embedded_vectors.append(np.squeeze(feature_vector))
        return embedded_vectors

    def similarity_score(self, img1_embeddings, img2_embeddings):
        """
        Calculates and returns cosine similarity scores.

        Parameters
        ----------
        img1_embeddings(list) : list of images embeddings
        img2_embeddings(list) : list of other images embeddings

        Returns
        -------
        sim_scores(list): cosine similarity scores
        """  

        sim_scores = []
        length = len(img1_embeddings)
        for idx in range(length):
            vec1 = img1_embeddings[idx]
            vec2 = img2_embeddings[idx]
            sim = 1 - distance.cosine(vec1, vec2)
            if np.isnan(sim):
                sim_scores.append(0)
            else:
                sim_scores.append(sim)
        return sim_scores

    def pearson_coeff(self, sim_scores, other_sim_scores):
        """
        Calculates and returns pearson coefficient score and returns it.

        Parameters
        ----------
        sim_scores(list) : list of cosine similarity scores
        other_sim_scores(list) : other list of cosine similarity scores

        Returns
        -------
        pearson_coefficient_score(numpy.float64): pearson coefficient score
        """  

        pearson_coefficient_score = round(pearsonr(sim_scores, other_sim_scores)[0], 2)
        return pearson_coefficient_score
