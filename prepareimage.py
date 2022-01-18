import numpy as np
import preprocess


def prepare(imagepath):
    processed_img = preprocess.preprocess(imagepath, img_w=128, img_h=64)
    x_arr = []
    x_arr.append(processed_img.T)
    arr = np.array(x_arr)
    processed_img = arr.reshape(1, 128, 64, 1)
    return processed_img