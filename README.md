# minor_project_5

Objective:  
+ To create embedding for similar images and dissimilar images to find similarity score 
+ To find latent representation of similar images and dissimilar images and compute similarity score
            
In this project, we have used VGG16 model, Mobile Net Model.

### Code Structure
```
|
│   .gitignore
│   config.yaml
│   Pipfile
│   Pipfile.lock
│   README.md
│
├───data
│   │   img_sim_dataset.csv
│   │
│   └───images
│           01.jpg
│           02.jpg
│           03.jpg
│           04.jpg
│           05.jpg
│           06.jpg
│           07.jpg
│           08.jpg
│           09.jpg
│           10.jpg
│           11.jpg
│           12.jpg
│           13.jpg
│           14.jpg
│           15.jpg
│           16.jpg
│           17.jpg
│           18.jpg
│           19.jpg
│           20.jpg
│           21.jpg
│           22.jpg
│           23.jpg
│           24.jpg
│           25.jpg
│           26.jpg
│           27.jpg
│           28.jpg
│           29.jpg
│           30.jpg
│
├───Notebook
│       image_similarity.ipynb
│
└───src
    │   main.py
    │   models.py
    │   pca.py
    │   vgg_model.py
    │
    └───__pycache__
            vgg_model.cpython-38.pyc
```

### Folders/files Description:
#### 1. Notebook
The Notebooks files are a quick way view the outputs.
- Contains image_similarity.ipynb file that shows the similarity and dissimilarity between the images using pre-trained VGG16 model

#### 2. Data
- Contains 30 RGB images distributed between 4 classes: Apple, Strawberry, Dogs and Cats


#### 3. src
- Contains the VGG_model.py which is a class built on pretrained VGG16 for image embedding and calculating the similarity score.
- Contains the models.py which is a class built on pretrained MobileNet Model for image embedding and calculating the similarity score.
- Contains the pca.py which is a class built to find feature vectors on dataset images using pca.
- Contains the main.py that calls the other 3 classes to perform image embedding and caluculate correlation score between images.

#### 4. Config.yaml
- has global variables, data paths and dictionaries


### How to run this repository:
Pre-requisits: Install pipenv(sudo apt-get pipenv)
#### Step 1: Setup 
- Clone the repository.
- pipenv shell
- setup the folder

#### Step 2.1: Run for Notebook
- you can run in jupyter Notebook locally or refer to remote notebook like colab,kaggle notebook etc

#### Step 3.2: Run for py file
- Run "python main.py"


### Output:
- Cosine Similarity between two images using vgg16.
- Pearson correlation score between two images using vgg16 and mobilenet.


