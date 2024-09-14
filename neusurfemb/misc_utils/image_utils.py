import copy
import numpy as np


def make_crop_square(x_min,
                     x_max,
                     y_min,
                     y_max,
                     min_valid_x=None,
                     max_valid_x=None,
                     min_valid_y=None,
                     max_valid_y=None,
                     scale_factor=1.0):

    def is_integer(x):
        return isinstance(x, int) or isinstance(x, np.integer)

    assert ((min_valid_x is None) == (max_valid_x is None))
    assert ((min_valid_y is None) == (max_valid_y is None))
    assert (is_integer(x_min) and is_integer(x_max) and is_integer(y_min) and
            is_integer(y_max))
    # Rescale the input bounds if required.
    assert (isinstance(scale_factor, float) and scale_factor > 0.0)

    # As a first step, if bounds are specified and on one of the two sides the
    # bounds are not fulfilled, adjust the input bounding box so as to fit
    # within them (which is assumed to be possible).
    exceeds = {'left': False, 'right': False, 'top': False, 'bottom': False}
    if (min_valid_x is not None and max_valid_x is not None):
        assert (min_valid_x < max_valid_x)
        assert (x_min <= max_valid_x and x_max >= min_valid_x)
        if (x_min < min_valid_x):
            assert (x_max <= max_valid_x)
            x_min = min_valid_x
            exceeds['left'] = True
        if (x_max > max_valid_x):
            assert (x_min >= min_valid_x)
            x_max = max_valid_x
            exceeds['right'] = True
    if (min_valid_y is not None and max_valid_y is not None):
        assert (min_valid_y < max_valid_y)
        assert (y_min <= max_valid_y and y_max >= min_valid_y)
        if (y_min < min_valid_y):
            assert (y_max <= max_valid_y)
            y_min = min_valid_y
            exceeds['top'] = True
        if (y_max > max_valid_y):
            assert (y_min >= min_valid_y)
            y_max = max_valid_y
            exceeds['bottom'] = True

    desired_length = max(x_max - x_min, y_max - y_min)

    assert (not ((exceeds['bottom'] and exceeds['top']) or
                 (exceeds['left'] and exceeds['right'])))

    if (exceeds['top']):
        # The object continues at the top -> Extend the crop towards the bottom
        # (padding if necessary).
        y_min_ = copy.deepcopy(y_min)
        y_max_ = y_max + (desired_length - (y_max - y_min))
    elif (exceeds['bottom']):
        # The object continues at the bottom -> Extend the crop towards the top
        # (padding if necessary).
        y_min_ = y_min - (desired_length - (y_max - y_min))
        y_max_ = copy.deepcopy(y_max)
    else:
        # The object is full visible in the vertical direction -> Equally
        # distribute the crop extension towards the bottom and the top.
        necessary_increase = desired_length - (y_max - y_min)
        y_min_ = y_min - necessary_increase // 2
        y_max_ = y_max + (necessary_increase - (y_min - y_min_))

    if (exceeds['left']):
        # The object continues at the left -> Extend the crop towards the right
        # (padding if necessary).
        x_min_ = copy.deepcopy(x_min)
        x_max_ = x_max + (desired_length - (x_max - x_min))
    elif (exceeds['right']):
        # The object continues at the right -> Extend the crop towards the left
        # (padding if necessary).
        x_min_ = x_min - (desired_length - (x_max - x_min))
        x_max_ = copy.deepcopy(x_max)
    else:
        # The object is full visible in the horizontal direction -> Equally
        # distribute the crop extension towards the left and the right.
        necessary_increase = desired_length - (x_max - x_min)
        x_min_ = x_min - necessary_increase // 2
        x_max_ = x_max + (necessary_increase - (x_min - x_min_))

    assert (x_max_ - x_min_ == y_max_ -
            y_min_), "Unable to make the given crop square."

    if (scale_factor != 1.0):
        x_center = (x_min_ + x_max_) / 2
        y_center = (y_min_ + y_max_) / 2.
        x_min_ = x_center + (x_min_ - x_center) * scale_factor
        x_max_ = x_center + (x_max_ - x_center) * scale_factor
        y_min_ = y_center + (y_min_ - y_center) * scale_factor
        y_max_ = y_center + (y_max_ - y_center) * scale_factor

        W_ = x_max_ - x_min_
        x_min_ = int(x_min_)
        x_max_ = x_min_ + W_
        H_ = y_max_ - y_min_
        y_min_ = int(y_min_)
        y_max_ = y_min_ + H_

    return int(x_min_), int(x_max_), int(y_min_), int(y_max_)
