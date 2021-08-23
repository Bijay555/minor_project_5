import logging
import yaml
import warnings
import pandas as pd
from src.vgg_model import VGG16_model


with open("config.yaml", 'r') as stream:
    config_file_instance = yaml.safe_load(stream)


if __name__ == "__main__":
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
    print("Pearson correlation value: ", score)
    
