from keras.layers import Input, Conv2D, Activation, Concatenate, BatchNormalization, Dense, Add, Dropout
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2
from keras.models import Model
from dlblocks import path, inception_block

def Inceptionv4(img_input, n_classes):

    def stem1_architecture():
        root = [('conv2_bn_relu',(3, 3), 32, (2, 2), 'valid'),
                ('conv2_bn_relu',(3, 3), 32, (1, 1), 'valid'),
                ('conv2_bn_relu',(3, 3), 64, (1, 1), 'same')]
        maxpool_branch = [('maxpool',(3, 3), (2, 2), 'valid')]
        maxpool_branch_architecture = {'root': maxpool_branch}
        convo_branch = [('conv2_bn_relu',(3, 3), 96, (2, 2), 'valid')]
        convo_branch_architecture = {'root': convo_branch}
        architecture = {'root': root, 'branches': [maxpool_branch_architecture,
                                                   convo_branch_architecture]}
        return architecture
    
    def stem2_architecture():
        convo_branch_1 = [('conv2_bn_relu', (1, 1), 64, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 96, (1, 1), 'valid')]
        convo_branch_1_architecture = {'root': convo_branch_1}
        convo_branch_2 = [('conv2_bn_relu', (1, 1), 64, (1, 1), 'same'),
                          ('conv2_bn_relu', (7, 1), 64, (1, 1), 'same'),
                          ('conv2_bn_relu', (1, 7), 64, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 96, (1, 1), 'valid')]
        convo_branch_2_architecture = {'root': convo_branch_2}
        architecture = {'branches': [convo_branch_1_architecture,
                                     convo_branch_2_architecture]}
        return architecture
    
    def stem3_architecture():
        convo_branch = [('conv2_bn_relu', (3, 3), 192, (2, 2), 'valid')]
        convo_branch_architecture = {'root': convo_branch}
        maxpool_branch = [('maxpool',(2, 2), (2, 2), 'valid')]
        maxpool_branch_architecture = {'root': maxpool_branch}
        architecture = {'branches': [convo_branch_architecture,
                                     maxpool_branch_architecture]}
        return architecture
    
    def blockA_architecture():
        convo_branch_1 = [('conv2_bn_relu', (1, 1), 96, (1, 1), 'same')]
        convo_branch_1_architecture = {'root': convo_branch_1}
        convo_branch_2 = [('conv2_bn_relu', (1, 1), 64, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 96, (1, 1), 'same')]
        convo_branch_2_architecture = {'root': convo_branch_2}
        convo_branch_3 = [('conv2_bn_relu', (1, 1), 64, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 96, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 96, (1, 1), 'same')]
        convo_branch_3_architecture = {'root': convo_branch_3}
        avgpool_branch = [('avgpool',(3, 3), (1, 1), 'same'),
                          ('conv2_bn_relu', (1, 1), 96, (1, 1), 'same')]
        avgpool_branch_architecture = {'root': avgpool_branch}
        architecture = {'branches': [convo_branch_1_architecture,
                                     convo_branch_2_architecture,
                                     convo_branch_3_architecture,
                                     avgpool_branch_architecture]}
        return architecture
    
    def blockB_architecture():
        convo_branch_1 = [('conv2_bn_relu', (1, 1), 384, (1, 1), 'same')]
        convo_branch_1_architecture = {'root': convo_branch_1}
        convo_branch_2 = [('conv2_bn_relu', (1, 1), 192, (1, 1), 'same'),
                          ('conv2_bn_relu', (1, 7), 224, (1, 1), 'same'),
                          ('conv2_bn_relu', (7, 1), 256, (1, 1), 'same')]
        convo_branch_2_architecture = {'root': convo_branch_2}
        convo_branch_3 = [('conv2_bn_relu', (1, 1), 192, (1, 1), 'same'),
                          ('conv2_bn_relu', (7, 1), 224, (1, 1), 'same'),
                          ('conv2_bn_relu', (1, 7), 224, (1, 1), 'same'),
                          ('conv2_bn_relu', (7, 1), 256, (1, 1), 'same'),
                          ('conv2_bn_relu', (1, 7), 256, (1, 1), 'same')]
        convo_branch_3_architecture = {'root': convo_branch_3}
        avgpool_branch = [('avgpool',(3, 3), (1, 1), 'same'),
                          ('conv2_bn_relu', (1, 1), 128, (1, 1), 'same')]
        avgpool_branch_architecture = {'root': avgpool_branch}
        architecture = {'branches': [convo_branch_1_architecture,
                                     convo_branch_2_architecture,
                                     convo_branch_3_architecture,
                                     avgpool_branch_architecture]}
        return architecture
    
    def blockC_architecture():
        convo_branch_1 = [('conv2_bn_relu', (1, 1), 256, (1, 1), 'same')]
        convo_branch_1_architecture = {'root': convo_branch_1}
        
        convo_branch_2_root = [('conv2_bn_relu', (1, 1), 384, (1, 1), 'same')]
        convo_branch_2_subbranch_1 = [('conv2_bn_relu', (1, 3), 256, (1, 1), 'same')]
        convo_branch_2_subbranch_1_architecture = {'root': convo_branch_2_subbranch_1}
        convo_branch_2_subbranch_2 = [('conv2_bn_relu', (3, 1), 256, (1, 1), 'same')]
        convo_branch_2_subbranch_2_architecture = {'root': convo_branch_2_subbranch_2}
        convo_branch_2_architecture = {'root': convo_branch_2_root,
                                       'branches': [convo_branch_2_subbranch_1_architecture,
                                                    convo_branch_2_subbranch_2_architecture]}
        
        convo_branch_3_root = [('conv2_bn_relu', (1, 1), 384, (1, 1), 'same'),
                               ('conv2_bn_relu', (3, 1), 448, (1, 1), 'same'),
                               ('conv2_bn_relu', (1, 3), 512, (1, 1), 'same')]
        convo_branch_3_subbranch_1 = [('conv2_bn_relu', (3, 1), 256, (1, 1), 'same')]
        convo_branch_3_subbranch_1_architecture = {'root': convo_branch_3_subbranch_1}
        convo_branch_3_subbranch_2 = [('conv2_bn_relu', (1, 3), 256, (1, 1), 'same')]
        convo_branch_3_subbranch_2_architecture = {'root': convo_branch_3_subbranch_2}
        convo_branch_3_architecture = {'root': convo_branch_3_root,
                                       'branches': [convo_branch_2_subbranch_1_architecture,
                                                    convo_branch_3_subbranch_2_architecture]}
        avgpool_branch = [('avgpool',(3, 3), (1, 1), 'same'),
                          ('conv2_bn_relu', (1, 1), 256, (1, 1), 'same')]
        avgpool_branch_architecture = {'root': avgpool_branch}
        architecture = {'branches': [convo_branch_1_architecture,
                                     convo_branch_2_architecture,
                                     convo_branch_3_architecture,
                                     avgpool_branch_architecture]}
        return architecture
    
    def reductionA_architecture():
        convo_branch_1 = [('conv2_bn_relu', (3, 3), 384, (2, 2), 'valid')]
        convo_branch_1_architecture = {'root': convo_branch_1}
        convo_branch_2 = [('conv2_bn_relu', (1, 1), 192, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 224, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 256, (2, 2), 'valid')]
        convo_branch_2_architecture = {'root': convo_branch_2}
        avgpool_branch = [('avgpool',(3, 3), (2, 2), 'valid')]
        avgpool_branch_architecture = {'root': avgpool_branch}
        architecture = {'branches': [convo_branch_1_architecture,
                                     convo_branch_2_architecture,
                                     avgpool_branch_architecture]}
        return architecture
    
    def reductionB_architecture():
        convo_branch_1 = [('conv2_bn_relu', (1, 1), 192, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 192, (2, 2), 'valid')]
        convo_branch_1_architecture = {'root': convo_branch_1}
        convo_branch_2 = [('conv2_bn_relu', (1, 1), 256, (1, 1), 'same'),
                          ('conv2_bn_relu', (1, 7), 256, (1, 1), 'same'),
                          ('conv2_bn_relu', (7, 1), 320, (1, 1), 'same'),
                          ('conv2_bn_relu', (3, 3), 320, (2, 2), 'valid')]
        convo_branch_2_architecture = {'root': convo_branch_2}
        maxpool_branch = [('maxpool',(3, 3), (2, 2), 'valid')]
        maxpool_branch_architecture = {'root': maxpool_branch}
        architecture = {'branches': [convo_branch_1_architecture,
                                     convo_branch_2_architecture,
                                     maxpool_branch_architecture]}
        return architecture
        
    # convolution args
    conv_args = {}
    conv_args['kernel_initializer'] = "he_normal"
    conv_args['kernel_regularizer'] = l2(1e-4)
    
    # stem part 1
    
    blocks = [
        stem1_architecture(),
        stem2_architecture(),
        stem3_architecture(),
        blockA_architecture(),
        blockA_architecture(),
        blockA_architecture(),
        blockA_architecture(),
        reductionA_architecture(),
        blockB_architecture(),
        blockB_architecture(),
        blockB_architecture(),
        blockB_architecture(),
        blockB_architecture(),
        blockB_architecture(),
        blockB_architecture(),
        reductionB_architecture(),
        blockC_architecture(),
        blockC_architecture(),
        blockC_architecture(),
    ]
    x = img_input
    for block_architecture in blocks:
        x = inception_block(x, block_architecture, conv_args)
        
    # classification
    x = AveragePooling2D((8, 8))(x)
    x = Dropout(0.8)(x)
    x = Flatten()(x)
    x = Dense(output_dim=n_classes, activation='softmax')(x)
    model = Model(img_input, x)
    return model

img_input = Input(shape=(299, 299, 3))
model = Inceptionv4(img_input, 1001)
model.summary()