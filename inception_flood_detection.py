import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np


#Initializing the dataset paths
path ='/content/drive/MyDrive/ALDA_project/Training_dataset/'
testing_path = '/content/drive/MyDrive/ALDA_project/test_set/'
img_path= path+"devset_images/"
label_path = path+ 'devset_images_gt.csv'
test_img_path= testing_path+"testset_images/"
test_label_path = testing_path+ 'testset_images_gt.csv'


#Formatting the input images
def append_ext(fn):
    return fn+".jpg"
traindf=pd.read_csv(label_path,dtype=str)
traindf.columns=(['id', 'label'])
traindf["id"]=traindf["id"].apply(append_ext)
testdf=pd.read_csv(test_label_path,dtype=str)
testdf.columns=(['id', 'label'])
testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.10)
test_datagen = ImageDataGenerator(rescale=1./255.)


#Creating the dataset generators for train, validation and test
train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory=img_path,
x_col="id",
y_col="label",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode='categorical',
target_size=(224,224))
valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory=img_path,
x_col="id",
y_col="label",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))
test_generator=datagen.flow_from_dataframe(
dataframe=testdf,
directory=test_img_path,
x_col="id",
y_col="label",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))



#Initiazling the image dimensions
img_height = 224
img_width = 224
IMG_SIZE = (img_height, img_width)



#Defining the pre-trained model
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255., offset= -1)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
image_batch, label_batch = next(iter(train_generator))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
print(label_batch.shape)
base_model.trainable = False


base_model.summary()


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


prediction_layer = tf.keras.layers.Dense(2)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])


inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)



base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()


#Training of the model
initial_epochs = 30
history = model.fit(train_generator,
                    epochs=initial_epochs,
                    validation_data=valid_generator)

#Fine-tuning the model
base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()



fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_generator,
                         epochs=total_epochs,
                         initial_epoch=10,
                         validation_data=valid_generator)


#Generating the predictions for test dataset
test_generator.reset()
y_pred = model.predict_generator(test_generator, verbose = True)
y_pred = np.argmax(y_pred, axis=1)


y_true = test_generator.classes


#Evaluation metrics for the test dataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_true, y_pred))
