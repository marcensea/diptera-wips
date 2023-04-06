"""
Definition of a MobileNet-inspired Keras models with 9 layers.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from os.path import join
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras import layers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta


def conv_block(x, filters, stride):
    """
    Definition of a block with depth-wise separable convolutions.
    :param x: the input tensor
    :param filters: number of convolution filters
    :param stride: stride of convolution, for image downsampling.
    :return: the output tensor
    """
    x = layers.SeparableConv2D(filters, (3, 3), strides=(stride, stride), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x


def build_model(input_shape, n_classes):

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same",
                     activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = conv_block(x, 64, 2)
    x = conv_block(x, 128, 2)
    x = layers.Dropout(0.25)(x)

    x = conv_block(x, 128, 2)
    x = layers.Dropout(0.25)(x)

    x = conv_block(x, 256, 2)
    x = layers.Dropout(0.25)(x)

    x = conv_block(x, 256, 1)
    x = layers.Dropout(0.25)(x)

    x = conv_block(x, 512, 1)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Categorization output
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model_compile(model)

    return model


def load_model():
    """
    Load a model architecture from JSON and load the weights from HDF5 file.
    """

    model_name = "mbn9_spp67_k1"
    model_path = join('..', '..', 'model_data', model_name)

    # Load a Keras models from a json file
    with open(join(model_path, model_name + '_model.json'), 'r') as input_file:
        model = model_from_json(input_file.read())
    print("Loaded model '%s'." % model_name)

    # Load weights
    model.load_weights(join(model_path, model_name + '_weights.hdf5'))

    return model


def model_compile(model):
    # Set metrics and optimizer
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(learning_rate=0.1),
                  # optimizer="Adadelta",
                  metrics=['categorical_accuracy'])


def replace_softmax(model, n_classes):
    """ Change the number of output classes in a model,
    by replacing the categorization layer. """
    layers = model.layers
    inputs = layers[0].input
    last_layer = model.layers[-2]  # last dense with relu activation
    print("Last layer selected:", last_layer.name)
    x = last_layer.output
    new_output = layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=new_output)
    return model


if __name__ == '__main__':

    model = load_model()
    model.summary()
    # replace_softmax(model)

