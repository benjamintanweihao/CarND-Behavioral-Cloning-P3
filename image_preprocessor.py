import cv2
import numpy as np
from random import choice, randint, uniform


def preprocess_image(image):
    cropped = image[80:134]
    resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
    blurred = blur(yuv)

    return blurred


def blur(image, kernel_size=5):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred


def flip_random(image, steering_angle):
    if randint(0, 1) == 0:
        image = np.fliplr(image)
        steering_angle *= -1.

    return image, steering_angle


def translate_random(image, steering_angle, range_x=50, range_y=10):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    return image, steering_angle


def brightness_random(image, steering_angle):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.5 + uniform(0, 1)
    image = np.array(image, dtype=np.float64)
    image[:, :, 2] *= random_bright
    image[:, :, 2][image[:, :, 2] > 255] = 255
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return image, steering_angle


def pipeline(image, steering_angle):
    image, steering_angle = translate_random(image, steering_angle)
    image, steering_angle = brightness_random(image, steering_angle)
    image, steering_angle = flip_random(image, steering_angle)

    return image, steering_angle


def generator(X, y, batch_size):
    X_batch = np.zeros(shape=(batch_size, 66, 200, 3), dtype=np.uint8)
    y_batch = np.zeros(shape=(batch_size, 1), dtype=np.float32)

    while True:
        for i in range(batch_size):
            index = choice(range(0, len(X)))
            X_batch[i], y_batch[i] = pipeline(X[index], y[index])

        yield X_batch, y_batch
