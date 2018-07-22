import csv          # to read the csv-file holding the driving data
import cv2
import numpy as np

lines = []          # array to store all the lines in the CSV-file
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []               # array with all the image file-names in the CSV-file (center, left and right)
steering_angles = []      # array with all the steering angle measurements in the CSV-file
for line in lines:
    for i in range(3):    # with looping 3 times we add the center, left and right images to the list of images
        source_path = line[i]
        filename = source_path.split('/')[-1]     # splits the filename as we only want the filename, not the path;
        current_path = 'data/IMG/' + filename
        # laad the image to store it in the list of images
        # Note that the simulator will feed RGB images to the trained model. Therefore we have to convert from BGR to RGB
        imgBGR = cv2.imread(current_path)
        imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
        images.append(imgRGB)
    # Read in the steering angle measurement and store it in the list of measurements
    # For the steering angles, for the center, we use the measured value, but for left and right we'll use a correction factor
    correction = 0.125
    st_ang = float(line[3])
    steering_angles.append(st_ang)                 # for the center image: use measured steering angle
    steering_angles.append(st_ang + correction)    # for left image: steer a bit more to the right
    steering_angles.append(st_ang - correction)    # for right image: steer a bit more to the left
print("All images and steering angles are read in!")

# Data augmentation
# As the track is counter clockwise, almost all steering is to the left. This causes left jerking. Therefore, we have to
# balance the data and for each image we have to create another flipped image steering to the right. Also the steering
# angle will be to the right in that case.
augm_images = []
augm_steering_angles = []
for img, st_ang in zip(images,steering_angles):
    augm_images.append(img)
    augm_steering_angles.append(st_ang)
    flipped_img = cv2.flip(img,1)      # '1' for flipping around vertical axle
    flipped_st_ang = st_ang * -1.0
    augm_images.append(flipped_img)
    augm_steering_angles.append(flipped_st_ang)

# feature array:
X_train = np.array(augm_images)
# label array:
y_train = np.array(augm_steering_angles)
print("The dataset has been augmented using flipped images from the original dataset!")

# Architecture selection: I have experimented with both the LeNet as the NVIDIA CNN architectures. Finally, the final
# model was trained using the NVIDIA network since that gave the best results. The model will predict the steering angle (label)
# based on the camera images fed as inputs (features). The adam optimizer is used and for the loss function, MSE (mean
# squared error) is used.
# To summarize: the input images are the features, and we have 1 output label which we try to predict: steering angle.

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

arch = 1
model = Sequential()
if arch == 0:                     # when arch is 0 we use LeNet, otherwise NVIDIA
    print('LeNet Arch used!')
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))    # normalize the data to -0.5 <> 0.5
    model.add(Cropping2D(cropping=((60,25),(0,0))))                            # crop 60/25 pixels off top/bottom
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
else:
    print('NVIDIA Arch used!')
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))    # normalize the data to -0.5 <> 0.5
    model.add(Cropping2D(cropping=((60,25),(0,0))))                            # crop 60/25 pixels off top/bottom
    model.add(Convolution2D(24,5,5,subsample=(2,2,),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2,),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2,),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

# What I want to do is minimize the error between the steering measurement which the network predicts and the ground
# truth steering measurement.
model.compile(loss='mse',optimizer='adam')

# After compiling the model its trained using the 'fit' KERAS function with feature/label arrays, shuffled and 20% of
# the data is used as a validation set. By default, Keras trains for 10 EPOCHS, but we see best results only using 3.
print("Training...")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

# Save the trained data so later it can be downloaded on the local machine to use in the simulator.
model.save('model.h5')
print("Trained model saved! To be used to drive autonomous in the simulator!")
 
