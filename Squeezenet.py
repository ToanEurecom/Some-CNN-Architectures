import numpy as np
from keras import backend as K
from keras.layers import Input, Conv2D, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, add
from keras.models import Model
from keras.engine.topology import Layer

def FireBlock(x, block_name, squeeze, expand):
    x = Conv2D(squeeze, (1, 1), padding='valid', name='squeeze' + '_' + block_name, activation="relu")(x)
    small_expand = Conv2D(expand, (1, 1), padding='valid', name='expand1x1' + '_' + block_name, activation="relu")(x)
    big_expand = Conv2D(expand, (3, 3), padding='same', name='expand3x3' + '_' + block_name, activation="relu")(x)
    x = concatenate([small_expand, big_expand], axis=3, name='concat_' + block_name)
    return x

def Squeezenet(img_input, classes):
    x = Conv2D(96, (3, 3), padding='Same', name='conv1')(img_input)

    x = FireBlock(x, block_name='fire2', squeeze=16, expand=64)
    x = FireBlock(x, block_name='fire3', squeeze=16, expand=64)
    x = FireBlock(x, block_name='fire4', squeeze=32, expand=128)
    x = FireBlock(x, block_name='fire5', squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

    x = FireBlock(x, block_name='fire6', squeeze=48, expand=192)
    x = FireBlock(x, block_name='fire7', squeeze=48, expand=192)
    x = FireBlock(x, block_name='fire8', squeeze=64, expand=256)
    x = FireBlock(x, block_name='fire9', squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)

    x = Convolution2D(10, (1, 1), padding='valid', name='conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    full_model = Model(img_input, x, name='squeezenet')
    return full_model