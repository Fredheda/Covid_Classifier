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
