#Covid-19 vs Viral Pneumonia vs Health Lung classifier for X-Ray images
#By Frederik Heda
#04/2021

#import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


#Assign Directories
DIRECTORY_train = "Covid_Data/Covid19-dataset/train"
DIRECTORY_test = "Covid_Data/Covid19-dataset/test"
DIRECTORY_predict = "Covid_Data/Covid19-dataset/Predict"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 16

#Creating Training Data
training_data_generator = ImageDataGenerator(rescale=1.0/255, zoom_range = 0.25, rotation_range = 20, width_shift_range = 0.05, height_shift_range = 0.05)
#training_data_generator = ImageDataGenerator(rescale = 1./255)
training_iterator = training_data_generator.flow_from_directory(DIRECTORY_train,class_mode=CLASS_MODE,color_mode=COLOR_MODE,batch_size=BATCH_SIZE)

training_iterator.next()

#Create Validation Data
validation_data_generator = ImageDataGenerator(rescale=1.0/255)
validation_iterator = validation_data_generator.flow_from_directory(DIRECTORY_test,class_mode=CLASS_MODE,color_mode=COLOR_MODE,batch_size=BATCH_SIZE)

#Create Prediction Data
prediction_data_generator = ImageDataGenerator(rescale=1.0/255)
prediction_iterator = validation_data_generator.flow_from_directory(DIRECTORY_predict,class_mode=CLASS_MODE,color_mode=COLOR_MODE,batch_size=1)
prediction_iterator.next()

#Create Model
model = Sequential(name = "Covid")
#Input Layer
model.add(layers.Input(shape=(256,256,1)))
#Add Hidden Layers
model.add(layers.Conv2D(5,5, strides = 2,padding="same", activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2), padding="valid"))
model.add(layers.Conv2D(3,3, strides = 1, padding="same", activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2),padding="valid"))

#Flatten Image:
model.add(layers.Flatten())

#Hidden Layer
model.add(layers.Dense(8,activation="relu"))
model.add(layers.Dense(5,activation="relu"))
#Output Layer
model.add(layers.Dense(3,activation = 'softmax'))

#Define Hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy()
metrics_function=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]

#Compile Model
model.compile(optimizer = optimizer, loss = loss_function, metrics = metrics_function)

#Print Model Summary
#print(model.summary())

model.fit(training_iterator, steps_per_epoch = training_iterator.samples/BATCH_SIZE, epochs = 30, validation_data = validation_iterator, validation_steps = validation_iterator.samples/BATCH_SIZE)

predictions = model.predict(prediction_iterator, verbose = 1)
predicted_classes = np.argmax(predictions, axis=1)

class_labels = ["Covid-19", "Normal", "Viral Pneumonia"]
class_labels_list = []
for i in range(len(predicted_classes)):
    if predicted_classes[i] == 0:
        class_labels_list.append(class_labels[0])
    elif predicted_classes[i] == 1:
        class_labels_list.append(class_labels[1])
    else:
        class_labels_list.append(class_labels[2])

image_list = []
listing = os.listdir("Covid_Data/Covid19-dataset/Predict/images_predict")    
for ImageFile in listing:
    if ".DS_Store" not in ImageFile:
        image_list.append(ImageFile)
    else:
        pass
image_list.sort()
print(image_list)
#print(image_list)
for i in range(len(image_list)):
    plt.figure()
    image = cv2.imread("Covid_Data/Covid19-dataset/Predict/images_predict/" + image_list[i])
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image)
    plt.title(class_labels_list[i])
    plt.show()

print(class_labels_list)
