from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D, Cropping2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import random

from image_preprocessor import preprocess_image


def populate_data(path_to_csv):
    images = []
    steering_angles = []

    with open(path_to_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            center_image_path = row[0]
            left_image_path = row[1]
            right_image_path = row[2]

            correction = 0.25
            center_steering_angle = float(row[3])

            left_steering_angle = center_steering_angle + correction
            right_steering_angle = center_steering_angle - correction

            center_img = preprocess_image(cv2.imread(center_image_path))
            left_img = preprocess_image(cv2.imread(left_image_path))
            right_img = preprocess_image(cv2.imread(right_image_path))

            images.append(center_img)
            steering_angles.append(center_steering_angle)

            images.append(left_img)
            steering_angles.append(left_steering_angle)

            images.append(right_img)
            steering_angles.append(right_steering_angle)

        return images, steering_angles


# 1. populate X_train and y_train

track_1_images, track_1_steering_angles = populate_data('data/track_1/driving_log.csv')

X = np.array(track_1_images)
y = np.array(track_1_steering_angles)

plt.hist(y, linestyle='solid')
plt.show()

# 2. create the model

model = Sequential()
model.add(Lambda(lambda image: image / 127.5, input_shape=(66, 200, 3), output_shape=(66, 200, 3)))
model.add(Conv2D(3, 5, strides=(1, 1), padding='valid', activation='elu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(24, 5, strides=(1, 1), padding='valid', activation='elu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(36, 5, strides=(1, 1), padding='valid', activation='elu'))
model.add(Conv2D(48, 3, strides=(1, 1), padding='valid', activation='elu'))
model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='elu'))
model.add(Flatten())
model.add(Dense(1164, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.summary()

# 3. compile the model

model.compile(optimizer=Adam(lr=0.0001), loss='mse')

# 4. fit the model

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=False)

# train_generator = datagen.flow(X_train, y_train, batch_size=32)

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=len(X_train),
#     epochs=3,
#     validation_data=(X_valid, y_valid))

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    validation_data=(X_valid, y_valid))

# 5. summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# 6. save the model

model.save('model.h5')

print('Model saved.')
