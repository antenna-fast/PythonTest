import cv2
import numpy as np


"""
corp image patch according to given bounding box [left_up_x, left_up_y, w, h]
"""
def corp_img(img, bbox):
    x, y, w, h = np.array(bbox, dtype=int)
    img_res = img[y: y+h, x:x+w]
    return img_res


