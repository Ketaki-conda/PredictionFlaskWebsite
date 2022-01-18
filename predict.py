from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as tf_keras_backend
import prepareimage
import numpy as np

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return tf_keras_backend.ctc_batch_cost(labels, y_pred, input_length,
                                           label_length)


def numbered_array_to_text(numbered_array):
    numbered_array = numbered_array[numbered_array != -1]
    letters = [
        ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A',
        'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
        'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
        'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    return "".join(letters[i] for i in numbered_array)


def predict(imagepath):
    image = prepareimage.prepare(imagepath)
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
    iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(
        iam_layers)  # (None,64, 32, 64)

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
    iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(
        iam_layers)  # (None, 32, 8, 256)

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
    iam_layers = layers.Bidirectional(
        layers.LSTM(units=256, return_sequences=True))(iam_layers)
    # layer nine
    iam_layers = layers.Bidirectional(
        layers.LSTM(units=256, return_sequences=True))(iam_layers)
    iam_layers = layers.BatchNormalization()(iam_layers)

    # transforms RNN output to character activations:
    iam_layers = layers.Dense(80,
                              kernel_initializer='he_normal',
                              name='dense2')(iam_layers)
    iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)

    labels = layers.Input(name='the_labels', shape=[16], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    # loss function
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1, ), name='ctc')(
        [iam_outputs, labels, input_length, label_length])
    iam_model_pred = Model(inputs=input_data, outputs=iam_outputs)
    iam_model_pred.load_weights(filepath='HandwrittenWordsModel-Try2.h5')

    test_predictions_encoded = iam_model_pred.predict(x=image)
    test_predictions_decoded = tf_keras_backend.get_value(
        tf_keras_backend.ctc_decode(
            test_predictions_encoded,
            input_length=np.ones(test_predictions_encoded.shape[0]) *
            test_predictions_encoded.shape[1],
            greedy=True)[0][0])
    return numbered_array_to_text(test_predictions_decoded[0])
