#This File is used to preprocess the image data and create relevant functions

#Assign Directories
DIRECTORY_train = "Covid_Data/Covid19-dataset/train"
DIRECTORY_test = "Covid_Data/Covid19-dataset/test"
DIRECTORY_predict = "Covid_Data/Covid19-dataset/Predict"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 25