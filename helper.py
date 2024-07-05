import pickle
from PIL import Image
import numpy as np
import cv2

model = pickle.load(open("./Model/model.pkl", "rb"))


def is_empty(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.array(image)/255
    img = cv2.resize(img, (28, 68))
    img = img.flatten().reshape(1, -1)
    pred = model.predict(img)

    if pred[0] == 0:
        return True
    else:
        return False


def get_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

def img_diff(img1, img2):
    return np.abs(np.mean(img1) - np.mean(img2))