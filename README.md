# multimodal-flood-classification

In this work we will focus on on-ground images taken by humans in the flooded regions and posted on social networks and therefore 
containing metadata. Initially, we will perform data pre-processing and exploratory data analysis on the flood images. (metadata_visualization.py)

## Dataset Description

The dataset for this task is curated for 2017 MediaEval Multimedia Satellite Task 1, "Disaster
Image Retrieval from Social Media" (DIRSM, link below). The dataset consists of the 6,600
(5,280 train/1,320 test) RGB Flickr images each pre-labeled with a “flooding”/”no flooding” and
accompanied by a set of metadata about the picture’s contents, time and location of creation, and
user-creator. Of the 6,600 total examples, 4,200 are labeled "no flooding" and 2,400 are labeled
"flooding". The training images are stored in a folder and the corresponding labels in ’.csv’ file. All
the images in the dataset were used for training and evaluation of the model is done on testing set
images of the DIRSM dataset.

## Image-only Model (inception_flood_detection.py and resnet_flood_detection.py)

### Inception V3: 
We have trained InceptionV3 pre-trained on Imagenet for flood
detection from images. The parameters used for training Inception V3 are:
 – Freezed all layers, except dense layer.
 – Epochs - 5
 – Learning rate - 0.0001
 – Adam Optimizer
 
### Resnet 50: 
We have trained Resnet 50 pre-trained on Imagenet for flood detection from
images. The parameters used for training Resnet 50 are:
 – Freezed all layers, except dense layer.
 – Epochs - 5
 – Learning rate - 0.0001
 – Adam Optimizer
 
## Metadata-only Model (binary_features.py and bert+svm_flood_detection.py)

### Binary Feature Extraction: 
Only three fields: “title”, “description”, and "user tags” in the
metadata are useful for labelling binary features. Flood-related keywords: flood, floods,
flooded, or flooding are used while flagging new binary features. Various machine learning
models were trained for the binary features. GridSearchCV was performed for each model to
achieve the best parameters.

### BERT + SVM:
Bidirectional Encoder Representations from Transformers (BERT), a deep
learning based approach which has given state of the art performance on various NLP
tasks.The BERT framework was pre-trained using text from Wikipedia. The embeddings
from BERT for "title" and "description" are concatenated and passed to SVM for classification.

## Multimodal (multimodal_flood_detection.py)

### IceptionV3 + BERT + Binary Feature Extraction:

We have built a multi modal approach
from the best achieving models with images and metadata, that is, we have combined
InceptionV3 giving the images as the input, Pre-trained BERT giving the metadata as input
and passed binary features extracted from the metadata to a fully connected network. For
each of the networks, batch normalization layer has improved the accuracy and decreased
the training time. The architecture of the multi modal approach has been shown in the
below.

![Screen Shot 2022-01-06 at 4 47 26 PM](https://user-images.githubusercontent.com/28782608/148456758-be7aad94-08a2-4009-b20d-295044ba4f2a.png)

