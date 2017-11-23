from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import choice

from image_preprocessor import preprocess_image, generator


def populate_data(path_to_csv):
    images = []
    steering_angles = []

    with open(path_to_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            center_image_path = row[0]
            left_image_path = row[1]
            right_image_path = row[2]

            correction = 0.20
            center_steering_angle = float(row[3])

            c = choice(range(3))

            if c == 0:
                image = preprocess_image(cv2.imread(center_image_path))
                steering_angle = center_steering_angle
            elif c == 1:
                image = preprocess_image(cv2.imread(left_image_path))
                steering_angle = center_steering_angle + correction
            else:
                image = preprocess_image(cv2.imread(right_image_path))
                steering_angle = center_steering_angle - correction

            images.append(image)
            steering_angles.append(steering_angle)

        return images, steering_angles


# 1. populate X_train and y_train

track_1_images, track_1_steering_angles = populate_data('data/track_1/driving_log.csv')

images = track_1_images
steering_angles = track_1_steering_angles


def resample_data(images, steering_angles):
    nb_bins = 25
    hist, bins = np.histogram(steering_angles, bins=nb_bins)
    avg_samples_per_bin = len(steering_angles) // nb_bins

    # compute keep_prob
    keep_probs = []
    threshold = avg_samples_per_bin * 0.5

    for i in range(nb_bins):
        if hist[i] < threshold:  # below avg
            keep_probs.append(1.0)
        else:
            keep_probs.append(1.0 / (hist[i] / threshold))

    # based on the frequency of the steering angle, compute indices to remove
    remove_indices = []

    for i in range(len(steering_angles)):
        for j in range(nb_bins):
            if bins[j] < steering_angles[i] <= bins[j + 1]:
                if np.random.rand() > keep_probs[j]:
                    remove_indices.append(i)

    X = np.delete(np.array(images), remove_indices, axis=0)
    y = np.delete(np.array(steering_angles), remove_indices)

    # plt.hist(y, bins=nb_bins)
    # plt.show()

    return X, y


X, y = resample_data(images, steering_angles)

# 2. create the model

model = Sequential()
model.add(Lambda(lambda image: image / 127.5 - 1, input_shape=(66, 200, 3), output_shape=(66, 200, 3)))
model.add(Conv2D(24, 5, strides=(2, 2), padding='valid', activation='elu'))
model.add(Conv2D(36, 5, strides=(2, 2), padding='valid', activation='elu'))
model.add(Conv2D(48, 5, strides=(2, 2), padding='valid', activation='elu'))
model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='elu'))
model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='elu'))
model.add(Flatten())
model.add(Dropout(0.50))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.summary()

# 3. compile the model

model.compile(optimizer=Adam(lr=0.0001), loss='mse')

# 4. fit the model

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, shuffle=True)

batch_size = 32

callbacks_list = [
    EarlyStopping(
        monitor='val_loss',
        patience=1
    ),
    ModelCheckpoint(
        'model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

history = model.fit_generator(
    generator(X_train, y_train, batch_size),
    steps_per_epoch=len(X_train),
    epochs=10,
    validation_data=generator(X_valid, y_valid, batch_size),
    validation_steps=100,
    callbacks=callbacks_list)

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
