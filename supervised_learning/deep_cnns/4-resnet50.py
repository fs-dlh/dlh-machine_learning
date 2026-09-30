#!/usr/bin/env python3
""" Builds the ResNet-50 architecture as described in
   Deep Residual Learning for Image Recognition (2015). """
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ Build the ResNet-50 model.

    Returns:
        The Keras model.
    """
    input_layer = K.Input(shape=(224, 224, 3))

    # Stage 0: initial convolution and max pooling
    X = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(input_layer)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(pool_size=(7, 7))(X)
    X = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(X)

    model = K.models.Model(inputs=input_layer, outputs=X)
    return model
