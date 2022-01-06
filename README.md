# multimodal-flood-classification

In this work we will focus on on-ground images taken by humans in the flooded regions and posted on social networks and therefore 
containing metadata. Initially, we will perform data pre-processing and exploratory data analysis on the flood images. (metadata_visualization.py)

#Dataset Description

The dataset for this task is curated for 2017 MediaEval Multimedia Satellite Task 1, "Disaster
Image Retrieval from Social Media" (DIRSM, link below). The dataset consists of the 6,600
(5,280 train/1,320 test) RGB Flickr images each pre-labeled with a “flooding”/”no flooding” and
accompanied by a set of metadata about the picture’s contents, time and location of creation, and
user-creator. Of the 6,600 total examples, 4,200 are labeled "no flooding" and 2,400 are labeled
"flooding". The training images are stored in a folder and the corresponding labels in ’.csv’ file. All
the images in the dataset were used for training and evaluation of the model is done on testing set
images of the DIRSM dataset.

#Image-only Model (inception_flood_detection.py and resnet_flood_detection.py)

Inception V3: We have trained InceptionV3 pre-trained on Imagenet for flood
detection from images. The parameters used for training Inception V3 are:
 – Freezed all layers, except dense layer.
 – Epochs - 5
 – Learning rate - 0.0001
 – Adam Optimizer
 
Resnet 50: We have trained Resnet 50 pre-trained on Imagenet for flood detection from
images. The parameters used for training Resnet 50 are:
 – Freezed all layers, except dense layer.
 – Epochs - 5
 – Learning rate - 0.0001
 – Adam Optimizer
 
