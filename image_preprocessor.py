import cv2


def preprocess_image(image):
    cropped = image[80:138]
    resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)

    return resized
