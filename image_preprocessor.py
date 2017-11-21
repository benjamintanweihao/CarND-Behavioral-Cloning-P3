import cv2
import numpy as np
from random import randint, uniform


def preprocess_image(image):
    cropped = image[80:134]
    resized = cv2.resize(cropped, (220, 66), interpolation=cv2.INTER_AREA)

    return resized


# augment

def flip_random(image, steering_angle):
    if randint(0, 1) == 0:
        image = np.fliplr(image)
        steering_angle *= 1.

    return image, steering_angle


# def translate_random(image, steering_angle):
#     rows, cols, _ = image.shape
#     trans_x = cols * uniform(-0.2, 0.2)
#     trans_y = rows * uniform(-0.2, 0.2)
#     M = np.float32([[1, 0, trans_x],  # left/right
#                     [0, 1, trans_y]])  # up/down
#
#     translated = cv2.warpAffine(image, M, (cols, rows))
#
#     return translated, steering_angle - trans_x * 0.004  # not sure


def translate_random(image, steering_angle, trans_range=200):
    # Translation
    rows, cols, _ = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steering_angle = steering_angle + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (cols, rows))

    return image_tr, steering_angle


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
    image = preprocess_image(image)

    return image, steering_angle

# path = 'data/track_1/IMG/center_2017_11_19_09_11_30_980.jpg'
# image = cv2.imread(path)
# augmented, _ = pipeline(image, 10)
# cv2.imshow('', augmented)
# cv2.waitKey(0)
