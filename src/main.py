import pandas as pd
from models import MobilenetEmbedding
from pca import PCAEmbeddings

models = ["MobilenetEmbedding"]
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
        pass
    
print("pearson_coeff_scores : ",pearson_coeff_scores)