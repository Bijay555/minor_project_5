import yaml
import logging
import warnings
import pandas as pd
from pca import PCAEmbeddings
from src.vgg_model import VGG16_model
from models import MobilenetEmbedding

models = ["MobilenetEmbedding", "VGG16_model"]
pearson_coeff_scores = {}

for model in models:
    if model == "MobilenetEmbedding":
        img_df = pd.read_csv("../data/img_sim_dataset.csv")
        mnEmb = MobilenetEmbedding("../data/images")
        
        img1_embeddings = mnEmb.get_feature_vector(img_df, "img1")
        img2_embeddings = mnEmb.get_feature_vector(img_df, "img2")
        
        sim_scores = mnEmb.similarity_score(img1_embeddings, img2_embeddings)
        pearson_coeff = mnEmb.pearson_coeff(sim_scores, img_df["sim"])
        pearson_coeff_scores[model] = pearson_coeff
        
        # Perform PCA and dimensionality reduction
        n_components = 70
        pcaemb_1 = PCAEmbeddings(img1_embeddings, n_components)
        pcaemb_2 = PCAEmbeddings(img2_embeddings, n_components)
        
        img1_embeddings_pca = pcaemb_1.transform()
        img2_embeddings_pca = pcaemb_2.transform()
        
        sim_scores = mnEmb.similarity_score(img1_embeddings_pca, img2_embeddings_pca)
        pearson_coeff = mnEmb.pearson_coeff(sim_scores, img_df["sim"])
        key = model + " with PCA"
        pearson_coeff_scores[key] = pearson_coeff
    else:
        with open("config.yaml", 'r') as stream:
            config_file_instance = yaml.safe_load(stream)
            
        image_path = Path(config_file_instance['image_dataset_path'])
        df = pd.read_csv(config_file_instance['dataset_path'])

        vgg_model = VGG16_model(image_path,df)
        df_img1_list = list(df['img1'])
        df_img2_list = list(df['img2'])   #images columns taken from dataframe

        assert len(df_img1_list) != 0, "No images found"
        assert len(df_img2_list) != 0, "No images found"

        logging.info("==== Generating Images Embedded Vectors====")
        feature_vector_list1 = vgg_model.feature_extractor(df_img1_list)
        feature_vector_list2 = vgg_model.feature_extractor(df_img2_list)

        logging.info("==== Finding Correlation between images using Pearson Correlation Cofficient ====")
        score = vgg_model.pearson_correlation(feature_vector_list1, feature_vector_list2)
        pearson_coeff_scores[model] = score
    
print("pearson_coeff_scores : ",pearson_coeff_scores)