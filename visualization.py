import cv2
import numpy as np

from config import COLORS

def colorize(mask):

    color_mask = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8
    )

    for cls, color in COLORS.items():
        color_mask[mask == cls] = color

    return color_mask

def overlay(image, color_mask):

    return cv2.addWeighted(
        image,
        0.6,
        color_mask,
        0.4,
        0
    )