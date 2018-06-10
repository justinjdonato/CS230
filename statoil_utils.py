"""General project specific utility functions"""

import cv2
import numpy as np


def rotate_image(img, angle=20):
    """Rotate image by a given degree"""

    # rotate image
    original = img.copy()

    M_rotate = cv2.getRotationMatrix2D((37, 37), angle, 1)
    img_new = cv2.warpAffine(img, M_rotate, (75, 75))

    length_row = 0
    length_column = 0
    boundary_step = 5

    for i in range(len(img_new)):
        if img_new[0, i] != float(0.0):
            length_row = i
            break
    for i in range(len(img_new)):
        if img_new[i, 0] != float(0.0):
            length_column = i
            break

    #  Substitute the padding from original image
    img_new[:length_column + boundary_step, :length_row + boundary_step] = original[:length_column + boundary_step,
                                                                           :length_row + boundary_step]
    img_new[-(length_row + boundary_step):, :length_column + boundary_step] = original[-(length_row + boundary_step):,
                                                                              :length_column + boundary_step]
    img_new[:length_row + boundary_step, -(length_column + boundary_step):] = original[:length_row + boundary_step,
                                                                              -(length_column + boundary_step):]
    img_new[-(length_column + boundary_step):, -(length_row + boundary_step):] = original[
                                                                                 -(length_column + boundary_step):,
                                                                                 -(length_row + boundary_step):]

    return img_new


def translate_horizontal(image, shift_horizontal=5):
    """Translate image horizontally by a shift"""

    # horizontally shift image
    img = image.copy()

    shift_vertical = 0;
    if shift_horizontal < 0:
        image_slice = img[:, shift_horizontal:].copy()
    if shift_horizontal > 0:
        image_slice = img[:, :shift_horizontal].copy()
    M_translate = np.float32([[1, 0, shift_horizontal], [0, 1, shift_vertical]])
    img_new = cv2.warpAffine(img, M_translate, (75, 75))

    # subsitute the padding from original image
    if shift_horizontal < 0:
        img_new[:, shift_horizontal:] = image_slice
    if shift_horizontal > 0:
        img_new[:, :shift_horizontal] = image_slice

    return img_new.reshape(75, 75).astype(np.float32)


def translate_vertical(image, shift_vertical=5):
    """Translate image vertically by a shift"""

    # vertically shift image
    img = image.copy()

    shift_horizontal = 0;
    if shift_vertical < 0:
        image_slice = img[shift_vertical:, :].copy()
    if shift_vertical > 0:
        image_slice = img[:shift_vertical, :].copy()
    M_translate = np.float32([[1, 0, shift_horizontal], [0, 1, shift_vertical]])
    img_new = cv2.warpAffine(img, M_translate, (75, 75))

    # subsitute the padding from original image
    if shift_vertical < 0:
        img_new[shift_vertical:, :] = image_slice
    if shift_vertical > 0:
        img_new[:shift_vertical, :] = image_slice

    return img_new.reshape(75, 75).astype(np.float32)


def translate_positive_diagonal(image, shift_diagonal=5):
    """Translate image along positive diagonal"""

    # translate image along positive diagonal
    img = image.copy()

    if shift_diagonal < 0:
        hor_slice = img[shift_diagonal:, :].copy()
        ver_slice = img[:, shift_diagonal:].copy()
    else:
        hor_slice = img[:shift_diagonal, :].copy()
        ver_slice = img[:, :shift_diagonal].copy()
    M_translate = np.float32([[1, 0, shift_diagonal], [0, 1, shift_diagonal]])
    img_new = cv2.warpAffine(img, M_translate, (75, 75))

    # substitute the padding from original image
    if shift_diagonal < 0:
        img_new[shift_diagonal:, :] = hor_slice
        img_new[:, shift_diagonal:] = ver_slice
    else:
        img_new[:shift_diagonal, :] = hor_slice
        img_new[:, :shift_diagonal] = ver_slice

    return img_new.reshape(75, 75).astype(np.float32)


def translate_negative_diagonal(image, shift_diagonal=5):
    """ Translate image along negative diagonal"""

    # translate image along negative diagonal
    img = image.copy()

    if shift_diagonal < 0:
        hor_slice = img[:-shift_diagonal, :].copy()
        ver_slice = img[:, shift_diagonal:].copy()
    if shift_diagonal > 0:
        hor_slice = img[-shift_diagonal:, :].copy()
        ver_slice = img[:, :shift_diagonal].copy()
    M_translate = np.float32([[1, 0, shift_diagonal], [0, 1, -shift_diagonal]])
    img_new = cv2.warpAffine(img, M_translate, (75, 75))

    # subsitute the padding from original image
    if shift_diagonal < 0:
        img_new[:-shift_diagonal, :] = hor_slice
        img_new[:, shift_diagonal:] = ver_slice
    if shift_diagonal > 0:
        img_new[-shift_diagonal:, :] = hor_slice
        img_new[:, :shift_diagonal] = ver_slice

    return img_new.reshape(75, 75).astype(np.float32)


def flip(image, direction=0):
    """Flip image"""

    img = image.copy()
    return cv2.flip(img, direction)


def zoom(image, zoom_shift=5):
    """Zoom image"""

    # zoom image
    img = image.copy()

    # zoom in
    if zoom_shift > 0:
        # scale
        img_new = cv2.resize(img, (75 + zoom_shift * 2, 75 + zoom_shift * 2))
        # crop
        img_new = img_new[zoom_shift:-zoom_shift, zoom_shift:-zoom_shift]
        # zoom out
    else:
        zoom_shift *= -1

        hor_top = img[:zoom_shift, :]
        hor_bottom = img[-zoom_shift:, :]
        ver_left = img[:, :zoom_shift]
        ver_right = img[:, -zoom_shift:]

        # scale
        img_new = cv2.resize(img, (75 - zoom_shift * 2, 75 - zoom_shift * 2))
        # zero padding
        img_new = cv2.copyMakeBorder(img_new, zoom_shift, zoom_shift, zoom_shift, zoom_shift,
                                     cv2.BORDER_CONSTANT, value=0.0)
        # Substitute the padding from original image
        img_new[:zoom_shift, :] = hor_top
        img_new[-zoom_shift:, :] = hor_bottom
        img_new[:, :zoom_shift] = ver_left
        img_new[:, -zoom_shift:] = ver_right

    return img_new.reshape(75, 75).astype(np.float32)



