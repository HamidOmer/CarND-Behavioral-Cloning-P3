import csv
import cv2
import numpy as np
import keras
import os
import sklearn
#import sklearn.model_selection.train_test_split
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from sklearn.utils import shuffle

# This works until after crossing the bridge and goes out of the lane at the turn and doesn't
# come back

lines = []
images = []
measurements = []

# Load Udacity Data
with open('./Udacity_Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
for line in lines:
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = './Udacity_Data/IMG/' + filename
#		print(local_path)
		image = cv2.imread(local_path)
#		cropped = image[25:90,:]
#		print ("before cropping:", image.shape)
#		print ("after cropping:", cropped.shape)
#		exit()
#		images.append(cropped)
		images.append(image)
	correction = 0.2
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement+correction)
	measurements.append(measurement-correction)

"""
# Load My Data
with open('./My_Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
for line in lines:
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = './My_Data/IMG/' + filename
		print(local_path)
		image = cv2.imread(local_path)
		#Crop image
		print (image.shape)
		exit()
		#cropimage = image[25:90,:]
		#images.append(cropimage)
	correction = 0.2
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement+correction)
	measurements.append(measurement-correction)
"""

augmented_images = []
augmented_measurements = []
for image,measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image,1)
	flipped_measurement = measurement * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

#augmented_images = np.array(augmented_images)
#augmented_measurements = np.array(augmented_measurements)

#Crop sky and bonnet
#for img in range(len(augmented_images)):
#augmented_images=augmented_images[:,25:90,:, :]

#print ("Shape of array before cropping:", augmented_images.shape)
#print ("Shape of array after cropping:", augmented_images.shape)
#print ("Shape of image after cropping:", augmented_images[0].shape)


######
def generator(lcr_images, str_angles, batch_size=32):
	num_samples = len(lcr_images)
	while 1: # Loop forever so the generator never terminates
		shuffle(lcr_images, str_angles)
		for offset in range(0, num_samples, batch_size):
			batch_images = lcr_images[offset:offset+batch_size]
			batch_angles = str_angles[offset:offset+batch_size]

# trim image to only see section with road
			X_train = np.array(batch_images)
			y_train = np.array(batch_angles)
			yield shuffle(X_train, y_train)

Xt_Images, Xv_Images, yt_angles, yv_angles = train_test_split(augmented_images, augmented_measurements, test_size=0.2, random_state=1)

print ("Total # of images:", len(augmented_images))
print ("Total # of steering angles:", len(augmented_measurements))
print ("Image shape:", Xt_Images[0].shape)
print ("# of Training samples:", len(Xt_Images))
print ("# of Validation samples:", len(Xv_Images))
#print (yt_angles.shape)
#print (yv_angles.shape)
	
# compile and train the model using the generator function
train_generator = generator(Xt_Images, yt_angles, batch_size=32)
validation_generator = generator(Xv_Images, yv_angles, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
#model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100, W_regularizer = l2(0.001)))
model.add(Dense(50, W_regularizer = l2(0.001)))
model.add(Dense(10, W_regularizer = l2(0.001)))
model.add(Dense(1, W_regularizer = l2(0.001)))


model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.fit_generator(train_generator, samples_per_epoch=len(Xt_Images), validation_data=validation_generator,nb_val_samples=len(Xv_Images), nb_epoch=3)
			
model.save('model.h5')


	