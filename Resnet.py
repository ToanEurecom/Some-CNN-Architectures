from keras.layers import Input, Conv2D, Activation, Concatenate, BatchNormalization, Dense, Add, Dropout
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2
from keras.models import Model
from dlblocks import path, shortcut

def pre_activation_Resnet(img_input, n_classes, residual_type, details):
    
    def bottleneck_architecture(filters, downsample):
        architecture = {}
        if downsample:
            root = [('bn_relu_conv2',(1, 1), filters, (2, 2), 'same'),
                    ('bn_relu_conv2',(3, 3), filters, (1, 1), 'same'),
                    ('bn_relu_conv2',(1, 1), filters*4, (1, 1), 'same')]
        else:
            root = [('bn_relu_conv2',(1, 1), filters, (1, 1), 'same'),
                    ('bn_relu_conv2',(3, 3), filters, (1, 1), 'same'),
                    ('bn_relu_conv2',(1, 1), filters*4, (1, 1), 'same')]
        architecture['root'] = root
        return architecture, False
    
    def naive_architecture(filters, downsample):
        architecture = {}
        if downsample:
            root = [('bn_relu_conv2',(3, 3), filters, (2, 2), 'same'),
                    ('bn_relu_conv2',(3, 3), filters, (1, 1), 'same')]
            
        else:
            root = [('bn_relu_conv2',(3, 3), filters, (1, 1), 'same'),
                    ('bn_relu_conv2',(3, 3), filters, (1, 1), 'same')]
        architecture['root'] = root
        return architecture, False
    
    if residual_type == 'naive':
        path_architecture = naive_architecture
    elif residual_type == 'bottleneck':
        path_architecture = bottleneck_architecture
    
    # convolution args
    conv_args = {}
    conv_args['kernel_initializer'] = "he_normal"
    conv_args['kernel_regularizer'] = l2(1e-4)
    downsample = False # first block does not need down sample
    
    # stem part
    x = Conv2D(64, (7, 7), strides=(2, 2),  padding="same")(img_input)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    
    # blocks part
    for repetitions, filters in details:
        for _ in range(repetitions):
            architecture, downsample = path_architecture(filters, downsample)
            conv_args['filters'] = filters
            x = shortcut(x, architecture, conv_args)
        downsample = True
        
    # classification part
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)
    
    model = Model(img_input, x)
    return model

def Resnet152(img_input, classes):
    #img_input = Input(shape=(224, 224, 3))
    details = [(3, 64),(8, 128),(36, 256),(3, 512)]
    model = pre_activation_Resnet(img_input, classes, 'bottleneck', details)
    return model