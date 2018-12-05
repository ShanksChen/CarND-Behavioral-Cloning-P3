import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.layers import Cropping2D
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split


# read the csv file
lines = []
with open("./training-data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# read the image file
images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split("/")[-1]
	current_path = "./training-data/IMG/" + filename
	image = cv2.imread(current_path)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement * -1.0)

# set the train array
# X_train = np.array(images)
# y_train = np.array(measurements)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=42)


# set the model LeNet-5
# model = Sequential()
# model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape = (160, 320, 3), output_shape = (160, 320, 3)))
# model.add(Cropping2D(cropping = ((50,20), (0,0))))
# model.add(Convolution2D(6, 5, 5, activation = "relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation = "relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120, activation = "relu"))
# model.add(Dropout(0.5))
# model.add(Dense(84, activation = "relu"))
# model.add(Dropout(0.5))
# model.add(Dense(1))

# set the model nvidia
model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape = (160, 320, 3), output_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((50,20), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss = "mse", optimizer = "adam")
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save("model.h5")
