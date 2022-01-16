from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as tf_keras_backend
import cv2
import numpy as np
import nltk

nltk.download('words')
from nltk.corpus import words
import difflib

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()

iam_model_pred = None
input_data = layers.Input(name='the_input',
                          shape=(128, 64, 1),
                          dtype='float32')  # (None, 128, 64, 1)

# Convolution layer (VGG)
iam_layers = layers.Conv2D(64, (3, 3),
                           padding='same',
                           name='conv1',
                           kernel_initializer='he_normal')(input_data)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(2, 2),
                                 name='max1')(iam_layers)  # (None,64, 32, 64)

iam_layers = layers.Conv2D(128, (3, 3),
                           padding='same',
                           name='conv2',
                           kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(iam_layers)

iam_layers = layers.Conv2D(256, (3, 3),
                           padding='same',
                           name='conv3',
                           kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.Conv2D(256, (3, 3),
                           padding='same',
                           name='conv4',
                           kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(1, 2),
                                 name='max3')(iam_layers)  # (None, 32, 8, 256)

iam_layers = layers.Conv2D(512, (3, 3),
                           padding='same',
                           name='conv5',
                           kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.Conv2D(512, (3, 3), padding='same',
                           name='conv6')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(iam_layers)

iam_layers = layers.Conv2D(512, (2, 2),
                           padding='same',
                           kernel_initializer='he_normal',
                           name='con7')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)

# CNN to RNN
iam_layers = layers.Reshape(target_shape=((32, 2048)),
                            name='reshape')(iam_layers)
iam_layers = layers.Dense(64,
                          activation='relu',
                          kernel_initializer='he_normal',
                          name='dense1')(iam_layers)

# RNN layer
# layer ten
iam_layers = layers.Bidirectional(layers.LSTM(
    units=256, return_sequences=True))(iam_layers)
# layer nine
iam_layers = layers.Bidirectional(layers.LSTM(
    units=256, return_sequences=True))(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)

# transforms RNN output to character activations:
iam_layers = layers.Dense(80, kernel_initializer='he_normal',
                          name='dense2')(iam_layers)
iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)

labels = layers.Input(name='the_labels', shape=[16], dtype='float32')
input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
label_length = layers.Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return tf_keras_backend.ctc_batch_cost(labels, y_pred, input_length,
                                           label_length)


# loss function
loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1, ), name='ctc')(
    [iam_outputs, labels, input_length, label_length])
iam_model_pred = Model(inputs=input_data, outputs=iam_outputs)

iam_model_pred.load_weights(filepath='HandwrittenWordsModel-Try2.h5')


def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    img_pad = np.ones([new_h, new_w, 3]) * 255
    img_pad[h1:h2, w1:w2, :] = img
    return img_pad


def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w < target_w and h < target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w >= target_w and h < target_h:
        new_w = target_w
        new_h = int(h * new_w / w)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w < target_w and h >= target_h:
        new_h = target_h
        new_w = int(w * new_h / h)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        ratio = max(w / target_w, h / target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img


def preprocess(path, img_w, img_h):
    img = cv2.imread(path)
    img = fix_size(img, img_w, img_h)
    ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img /= 255
    return img


def numbered_array_to_text(numbered_array):
    # count = 0
    if (type(numbered_array) != type(" ")):
        numbered_array = numbered_array[numbered_array != -1]

        letters = [
            ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.',
            '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
            '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
            'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z'
        ]
        return "".join(letters[i] for i in numbered_array)


def predict(imagepath):
    processed_img = preprocess(imagepath, img_w=128, img_h=64)
    x_arr = []
    x_arr.append(processed_img.T)
    arr = np.array(x_arr)
    processed_img = arr.reshape(1, 128, 64, 1)
    test_predictions_encoded = iam_model_pred.predict(x=processed_img)
    test_predictions_decoded = tf_keras_backend.get_value(
        tf_keras_backend.ctc_decode(
            test_predictions_encoded,
            input_length=np.ones(test_predictions_encoded.shape[0]) *
            test_predictions_encoded.shape[1],
            greedy=True)[0][0])
    text = numbered_array_to_text(test_predictions_decoded[0])
    word_list = words.words()
    wordsab = difflib.get_close_matches(text[2:], word_list)
    print(text)
    return text[2:]
    # if (len(wordsab) > 0):
    #     return wordsab[0]
    # else:
    #     return ""


predict("words/34.png")