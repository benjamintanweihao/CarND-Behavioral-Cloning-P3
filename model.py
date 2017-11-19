from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D, Cropping2D

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. read csv

def populate_data(path_to_csv):
    images = []
    steering_angles = []

    with open(path_to_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            center_image_path = row[0]
            left_image_path = row[1]
            right_image_path = row[2]

            correction = 0.4
            center_steering_angle = float(row[3])
            left_steering_angle = center_steering_angle + correction
            right_steering_angle = center_steering_angle - correction

            center_img = cv2.imread(center_image_path)
            left_img = cv2.imread(left_image_path)
            right_img = cv2.imread(right_image_path)

            # center

            images.append(center_img)
            steering_angles.append(center_steering_angle)

            # flip image and steering angle

            images.append(np.fliplr(center_img))
            steering_angles.append(-center_steering_angle)

            # left

            images.append(left_img)
            steering_angles.append(left_steering_angle)

            # flip image and steering angle

            images.append(np.fliplr(left_img))
            steering_angles.append(-left_steering_angle)

            # right

            images.append(right_img)
            steering_angles.append(float(right_steering_angle))

            # flip image and steering angle

            images.append(np.fliplr(right_img))
            steering_angles.append(-(float(right_steering_angle)))

        return images, steering_angles


# 2. populate X_train and y_train

track_1_images, track_1_steering_angles = populate_data('data/track_1/driving_log.csv')
track_1_opp_images, track_1_opp_steering_angles = populate_data('data/track_1_opp/driving_log.csv')

X_train = np.array(track_1_images + track_1_opp_images)
y_train = np.array(track_1_steering_angles + track_1_opp_steering_angles)

# 3. create the model

model = Sequential()
model.add(Lambda(lambda image: image / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(6, 5, strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(6, 5, strides=(1, 1), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.75))
model.add(Dense(84))
model.add(Dropout(0.75))
model.add(Dense(1))

model.summary()

# 4. compile the model
model.compile(optimizer='adam', loss='mse')

# 5. fit the model

history = model.fit(X_train, y_train, epochs=2, validation_split=0.2, shuffle=True)

# 6. summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# 7. save the model

model.save('model.h5')

print('Model saved.')
