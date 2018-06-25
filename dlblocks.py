import numpy as np
from keras import backend as k
from keras.layers import Input, Conv2D, Activation, Concatenate, BatchNormalization, Dense, Add, Dropout
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.models import Model
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras.models import Model

def FireBlock(x, block_name, squeeze, expand):
    
    x = Conv2D(squeeze, (1, 1), padding='valid', name='squeeze' + '_' + block_name, activation="relu")(x)
    small_expand = Conv2D(expand, (1, 1), padding='valid', name='expand1x1' + '_' + block_name, activation="relu")(x)
    big_expand = Conv2D(expand, (3, 3), padding='same', name='expand3x3' + '_' + block_name, activation="relu")(x)
    x = concatenate([small_expand, big_expand], axis=3, name='concat_' + block_name)
    return x

def bn_relu_conv2(x, conv_args):
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=conv_args['filters'],
                   kernel_size=conv_args['kernel_size'],
                   strides=conv_args['strides'],
                   padding=conv_args['padding'],
                   kernel_initializer=conv_args['kernel_initializer'],
                   kernel_regularizer=conv_args['kernel_regularizer'])(x)
    return x

def conv2_bn_relu(x, conv_args):
    x = Conv2D(filters=conv_args['filters'],
               kernel_size=conv_args['kernel_size'],
               strides=conv_args['strides'],
               padding=conv_args['padding'],
               kernel_initializer=conv_args['kernel_initializer'],
               kernel_regularizer=conv_args['kernel_regularizer'])(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x) 
    return x

def path(x, architecture, conv_args):
    
    if 'root' in architecture:
        root = architecture['root']
        for layer in root:
            if layer[0] == 'bn_relu_conv2':
                _, kernel_size, filters, strides, padding, = layer
                conv_args['kernel_size'] = kernel_size
                conv_args['filters'] = filters
                conv_args['strides'] = strides
                conv_args['padding'] = padding
                x = bn_relu_conv2(x, conv_args)
                #if filters == 1024:
                #    print(filters)
                
            if layer[0] == 'conv2_bn_relu':
                _, kernel_size, filters, strides, padding, = layer
                conv_args['kernel_size'] = kernel_size
                conv_args['filters'] = filters
                conv_args['strides'] = strides
                conv_args['padding'] = padding
                x = conv2_bn_relu(x, conv_args)
                
            elif layer[0] == 'maxpool':
                _, pool_size, strides, padding, = layer
                x = MaxPooling2D(pool_size=pool_size,
                                 strides=strides,
                                 padding=padding)(x)
            elif layer[0] == 'avgpool':
                _, pool_size, strides, padding, = layer
                x = AveragePooling2D(pool_size=pool_size,
                                     strides=strides,
                                     padding=padding)(x)
                
    xs = []
    if 'branches' in architecture:
        branch_architectures = architecture['branches']
        for branch_architecture in branch_architectures:
            xs += path(x, branch_architecture, conv_args)
    else:
        xs.append(x)
            
    return xs

def shortcut(x, architecture, conv_args):
    
    # 
    x_residual = path(x, architecture, conv_args)
    x_residual = x_residual[0] 
    
    # check size of x_residual and x
    x_residual_shape = k.int_shape(x_residual)
    x_shape = k.int_shape(x)
    
    stride_width = int(round(x_shape[1] / x_residual_shape[1]))
    stride_height = int(round(x_shape[2] / x_residual_shape[2]))
    equal_channels = x_shape[3] == x_residual_shape[3]
    
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        x_shortcut = Conv2D(filters=x_residual_shape[3],
                            kernel_size=(1,1),
                            strides=(stride_width, stride_height),
                            padding='same',
                            kernel_initializer=conv_args['kernel_initializer'],
                            kernel_regularizer=conv_args['kernel_regularizer'])(x)
    else:
        x_shortcut = x
 
    return Add()([x_residual, x_shortcut])

def inception_block(x, architecture, conv_args, output_size = None):  
    #
    xs = path(x, architecture, conv_args)
    x = Concatenate(axis=-1)(xs)
    #   
    if output_size:
        conv_args = layers_args['conv_args']
        x = Conv2D(filters=x_residual_shape[3],
                   kernel_size=(1,1),
                   padding=conv_args['padding'],
                   kernel_initializer=conv_args['kernel_initializer'],
                   kernel_regularizer=conv_args['kernel_regularizer'])(x)
    
    return x
