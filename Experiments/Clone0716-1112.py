import csv
import cv2
import numpy as np
import keras
import os
import sklearn
import math
import random
import matplotlib.pyplot as plt
#import sklearn.model_selection.train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, Lambda, Cropping2D, Input, ELU
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from sklearn.utils import shuffle

# Works on track 1 under 15mph. Did not test beyond 15mph except for 30mph where it fails
# Useful functions inspired by Vivek (https://github.com/vxy10/P3-BehaviorCloning/blob/master/model.py)
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
	
def trans_image(image,steer,trans_range):
    # Translation
    rows, cols, _ = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang,tr_x
	
def preprocess_image(image, steering):
    # Preprocessing training files and augmenting
	'''
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path_file = line_data['left'][0].strip()
        shift_ang = .25
    if (i_lrc == 1):
        path_file = line_data['center'][0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = line_data['right'][0].strip()
        shift_ang = -.25
    y_steer = line_data['steer_sm'][0] + shift_ang
    image = cv2.imread(path_file)
	'''
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	image,steering,tr_x = trans_image(image,steering,150)
	image = augment_brightness_camera_images(image)
    #image = np.array(image)
	#Flip random images#
	ind_flip = np.random.randint(2)
	if ind_flip==0:
		image = cv2.flip(image,1)
	#flip_coin = random.randint(0,1)
	#if flip_coin == 1:
	#	image = cv2.flip(image,1)
		steering = -steering
	return image, steering

'''
def random_shadows(image):
    """
    Generate a random shadow on the image
    area parameter is a percentage of the total image area
    """
    shadows = image.copy()

    image_area = shadows.shape[0] * shadows.shape[1]
    poly = get_shadow_poly(shadows.shape[0], shadows.shape[1])
    cv2.fillPoly(shadows, np.array([poly]), -1)

    alpha = np.random.uniform(0.6, 0.9)
    return cv2.addWeighted(shadows, alpha, image, 1-alpha,0,image)

def get_shadow_poly(max_x, max_y):
    """
    Get the polygons of a random area delimited between
    max_x, max_y. The polygons generated will either be horizontal
    or vertically aligned with the image, with a random distribution.
    """
    horizontal = np.random.uniform()

    if horizontal < 0.5:
        x1 = random.randint(0, max_x/2)
        y1 = 0
        x2 = random.randint(max_x / 2, max_x)
        y2 = 0
        x3 = random.randint(max_x / 2, max_x)
        y3 = max_y       
        x4 = random.randint(0, max_x/2)
        y4 = max_y
    else:
        x1 = 0
        y1 = random.randint(0, max_y/2)
        x2 = 0
        y2 = random.randint(max_y / 2, max_y)
        x3 = max_x
        y3 = random.randint(max_y / 2, max_y)       
        x4 = max_x
        y4 = random.randint(0, max_y/2)
    
    return [[x1,y1],[x2,y2],[x3,y3], [x4,y4]]	
'''
	
lines_ud = []
images = []
steerings = []
correction = 0.2

# Load Udacity Data
with open('./Udacity_Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines_ud.append(line)
		
for line in lines_ud:
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = './Udacity_Data/IMG/' + filename
#		print(local_path)
		image = cv2.imread(local_path)
#		cv2.imshow("Org img", image)
		cropped = image[65:135,:,:]
#		cv2.imshow("Cropped img", cropped)
#		cv2.waitKey(0)
#		exit()
#		print ("before cropping:", image.shape)
#		print ("after cropping:", cropped.shape)
#		exit()
		images.append(cropped)
#		images.append(image)
	steering = float(line[3])
	steerings.append(steering)
	steerings.append(steering+correction)
	steerings.append(steering-correction)

lines_my = []
# Load My Data
with open('./My_Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines_my.append(line)
		
for line in lines_my:
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = './My_Data/IMG/' + filename
		#print(local_path)
		image = cv2.imread(local_path)
		cropped = image[65:135,:]
		#Crop image
		#print (image.shape)
		#exit()
		#cropimage = image[25:90,:]
		images.append(cropped)
	steering = float(line[3])
	steerings.append(steering)
	steerings.append(steering+correction)
	steerings.append(steering-correction)

images = np.array(images)
steerings = np.array(steerings)
'''
augmented_images = []
augmented_steerings = []
for image,steering in zip(images, steerings):
	augmented_images.append(image)
	augmented_steerings.append(steering)
	flipped_image = cv2.flip(image,1)
	flipped_steering = steering * -1.0
	augmented_images.append(flipped_image)
	augmented_steerings.append(flipped_steering)

#augmented_images = np.array(augmented_images)
#augmented_steerings = np.array(augmented_steerings)

#Crop sky and bonnet
#for img in range(len(augmented_images)):
#augmented_images=augmented_images[:,25:90,:, :]

#print ("Shape of array before cropping:", augmented_images.shape)
#print ("Shape of array after cropping:", augmented_images.shape)
#print ("Shape of image after cropping:", augmented_images[0].shape)
'''

######
'''
pimages = []
psteerings = []
def generator(lcr_images, str_angles, batch_size=64):
	num_samples = len(lcr_images)
	while 1: # Loop forever so the generator never terminates
		shuffle(lcr_images, str_angles)
		for offset in range(0, num_samples, batch_size):
			batch_images = lcr_images[offset:offset+batch_size]
			batch_steerings = str_angles[offset:offset+batch_size]
			for image,steering in zip(batch_images, batch_steerings):
				image, steering = preprocess_image(image, steering)
				pimages.append(image)
				psteerings.append(steering)
				#print(pimages[0])
				#print(psteerings[0])
				#exit()
# trim image to only see section with road
			X = np.array(pimages)
			y = np.array(psteerings)
			yield shuffle(X, y)
'''
pr_thr = 1
batchsz = 64

def generator(lcr_images, str_angles, batch_size=batchsz):
	batch_images = np.zeros((batch_size, 70, 320, 3))
	batch_steerings = np.zeros(batch_size)
#	batch_images = []
#	batch_steerings = []
	while True: # Loop forever so the generator never terminates
		shuffle(lcr_images, str_angles)
		for i in range(batch_size):
			i_rand = np.random.randint(len(lcr_images)-1)
			#pimg = lcr_images.iloc[[i_rand]].reset_index()
			#pstr = lcr_str_angles.iloc[[i_rand]].reset_index()

			keep_pr = 0
			while keep_pr == 0:
				#X, y = preprocess_image(pimg, pstr)
				X, y = preprocess_image(lcr_images[i_rand], str_angles[i_rand])

				if abs(y)<0.15:
					pr = np.random.uniform()
					if pr>pr_thr:
						keep_pr = 1
				else:
					keep_pr = 1
			
			batch_images[i] = X
			batch_steerings[i] = y

		yield batch_images,batch_steerings
			
def val_generator(lcr_images, str_angles, batch_size=batchsz):
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
			
Xt_Images, Xv_Images, yt_angles, yv_angles = train_test_split(images, steerings, test_size=0.2, random_state=1)

print ("Total # of images:", len(images))
print ("Total # of steering angles:", len(steerings))
print ("Image shape:", Xt_Images[0].shape)
print ("# of Training samples:", len(Xt_Images))
print ("# of Validation samples:", len(Xv_Images))
#print (yt_angles.shape)
#print (yv_angles.shape)
#exit()
	
# compile and train the model using the generator function
#train_generator = generator(Xt_Images, yt_angles, batch_size=batchsz)
#print ("Size of training generator images array:", len(train_generator))
#imtmp, strmp = train_generator
#print ("Generator output shape:", train_generator.shape)
#print ("Number of images in batch:", len(imtmp))
#print ("Number of steerings in batch:", len(strmp))
#exit()

#validation_generator = val_generator(Xv_Images, yv_angles, batch_size=batchsz)
# Nvidia model

input_shape = Xt_Images[0].shape
#print ("Processed Image shape:", train_generator[0][0].shape)
model = Sequential()
#model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
#model.add(Lambda(lambda x: x/255.0 - 0.5),input_shape=(70,320,3))
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=input_shape))
#model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Dropout(.3))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Dropout(.3))
model.add(Flatten())
model.add(Dense(1164, W_regularizer = l2(0.001)))
model.add(Dropout(.5))
model.add(Dense(100, W_regularizer = l2(0.001)))
model.add(Dropout(.5))
model.add(Dense(50, W_regularizer = l2(0.001)))
model.add(Dropout(.5))
model.add(Dense(10, W_regularizer = l2(0.001)))
model.add(Dropout(.5))
model.add(Dense(1))

'''
#Vivek's model
input_shape = Xt_Images[0].shape
filter_size = 3
pool_size = (2,2)
model = Sequential()
model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))
model.add(Conv2D(3,1,1,
                        border_mode='valid',
                        name='conv0', init='he_normal'))
model.add(Conv2D(32,filter_size,filter_size,
                        border_mode='valid',
                        name='conv1', init='he_normal'))
model.add(ELU())
model.add(Conv2D(32,filter_size,filter_size,
                        border_mode='valid',
                        name='conv2', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Conv2D(64,filter_size,filter_size,
                        border_mode='valid',
                        name='conv3', init='he_normal'))
model.add(ELU())

model.add(Conv2D(64,filter_size,filter_size,
                        border_mode='valid',
                        name='conv4', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Conv2D(128,filter_size,filter_size,
                        border_mode='valid',
                        name='conv5', init='he_normal'))
model.add(ELU())
model.add(Conv2D(128,filter_size,filter_size,
                        border_mode='valid',
                        name='conv6', init='he_normal'))
model.add(ELU())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512,name='hidden1', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(64,name='hidden2', init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(16,name='hidden3',init='he_normal'))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(1, name='output', init='he_normal'))
'''

model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
#model.fit_generator(train_generator, samples_per_epoch=(len(Xt_Images)//batchsz)*batchsz, validation_data=validation_generator,nb_val_samples=len(Xv_Images), nb_epoch=4)
			
#model.save('model.h5')
from pathlib import Path
import json

def save_model(fileModelJSON,fileWeights):
    #print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)
	

best_iter = 0
best_vloss = 50

for i in range(5):

    train_generator = generator(Xt_Images, yt_angles, batch_size=batchsz)
    validation_generator = val_generator(Xv_Images, yv_angles, batch_size=batchsz)

    history = model.fit_generator(train_generator, samples_per_epoch=(len(Xt_Images)//batchsz)*batchsz, 
            nb_epoch=1, validation_data=validation_generator, nb_val_samples=len(Xv_Images))
    
    fileModelJSON = 'model_' + str(i) + '.json'
    fileWeights = 'model_' + str(i) + '.h5'
    
    save_model(fileModelJSON,fileWeights)
    
    val_loss = history.history['val_loss'][0]
    if val_loss < best_vloss:
        best_iter = i 
        best_vloss = val_loss
        fileModelJSON = 'model_best.json'
        fileWeights = 'model_best.h5'
        save_model(fileModelJSON,fileWeights)
        
    pr_thr = 1/(i+1)
print('Best model found at iteration # ' + str(best_iter))
print('Best validation loss value: ' + str(np.round(best_vloss,4)))

	