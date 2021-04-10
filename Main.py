#Covid-19 vs Viral Pneumonia vs Health Lung classifier for X-Ray images
#By Frederik Heda
#09/04/2021

#import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2

#Assign Directories
DIRECTORY_train = "Covid_Data/Covid19-dataset/train"
DIRECTORY_test = "Covid_Data/Covid19-dataset/test"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 32

#Creating Training Data
training_data_generator = ImageDataGenerator(1.0/255, zoom_range = 0.05, rotation_range = 20, width_shift_range = 0.02, height_shift_range = 0.02)

training_iterator = training_data_generator.flow_from_directory(DIRECTORY_train,class_mode=CLASS_MODE,color_mode=COLOR_MODE,batch_size=BATCH_SIZE)

training_iterator.next()

#Create Validation Data
validation_data_generator = ImageDataGenerator()
validation_iterator = validation_data_generator.flow_from_directory(DIRECTORY_test,class_mode=CLASS_MODE,color_mode=COLOR_MODE,batch_size=BATCH_SIZE)

#Create Model
model = Sequential(name = "Covid")
#Input Layer
model.add(layers.Input(shape=(256,256,1)))
#Add Hidden Layers
model.add(layers.Conv2D(5,5, strides = 2, activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(layers.Conv2D(3,3, strides = 1, activation = 'relu'))

#Flatten Image:
model.add(layers.Flatten())
#Output Layer
model.add(layers.Dense(3,activation = 'softmax'))

#Define Hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
loss_function = tf.keras.losses.CategoricalCrossentropy()
metrics_function=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]

#Compile Model
model.compile(optimizer = optimizer, loss = loss_function, metrics = metrics_function)

#Print Model Summary
#print(model.summary())