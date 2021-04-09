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

#Assign Directories
DIRECTORY_train = "Covid_data/train"
DIRECTORY_test = "Covid_data/test"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 32