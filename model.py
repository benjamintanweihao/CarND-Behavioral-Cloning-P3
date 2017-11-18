from keras.models import Sequential
from keras.layers import Dense, Flatten

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


images = []
steering_angles = []

# 1. read csv

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    for row in reader:
        center_image_path, steering_angle = row[0], row[3]

        img = cv2.imread(center_image_path)

        images.append(img)
        steering_angles.append(float(steering_angle))

# 2. populate X_train and y_train

X_train = np.array(images)
y_train = np.array(steering_angles)

# 3. create the model

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

# 4. compile the model
model.compile(optimizer='adam', loss='mse')

# 5. fit the model

history = model.fit(X_train, y_train, nb_epoch=10, validation_split=0.2, shuffle=True)

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