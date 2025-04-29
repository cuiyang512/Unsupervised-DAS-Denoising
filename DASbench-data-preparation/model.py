# Build Channel Attention Block
from keras.layers import *
from keras.utils import Sequence
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import math

def eca_block(tensor, b=1, gama=8):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    eca_shape = (1,filters)
    init1 = Reshape(target_shape=(eca_shape))(init)   
    in_channel=1
    
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
    
    if kernel_size % 2:
        kernel_size = kernel_size
    
    else:
        kernel_size = kernel_size + 1
    
    x = GlobalAveragePooling1D()(init1)
    x = Reshape(eca_shape)(x)

    x = Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x) 
    print('k size 1:', kernel_size)
    x = tf.nn.relu(x)
    x = Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)  
    print('k size 2:', kernel_size)
    x = tf.nn.sigmoid(x)    
    x = Reshape((1,1,in_channel))(x)
    #x = GlobalAveragePooling1D()(x)
    
    outputs = multiply([init1, x])
    outputs = tf.keras.backend.squeeze(outputs,1)
    # outputs = GlobalAveragePooling1D()(outputs)
    return outputs


def inception_res(input_tensor):
    e0 = input_tensor

    e1 = Conv1D(1, 1, padding='same')(e0)
    e1 = Conv1D(1, 3, padding='same', activation='tanh')(e1)
    e1 = Conv1D(1, 3, padding='same', activation='tanh')(e1)
    e1 = BatchNormalization()(e1)
    
    e2 = Conv1D(1, 1, padding='same')(e0)
    e2 = Conv1D(1, 3, padding='same', activation='tanh')(e2)
    e2 = BatchNormalization()(e2)

    e3 = Conv1D(1, 1, padding='same')(e0)
    e3 = BatchNormalization()(e3)
    
    con = Concatenate()([e1, e2, e3])
    con = Conv1D(1, 1, padding='same')(con)

    con_1 = Add()([con, e0])
    # out_tensor = tf.nn.PReLU(con_1)

    return(con_1)

def mff_block(input_tensor, filters, kernel_size, strides):
    e1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    e1 = BatchNormalization()(e1)
    e1 = PReLU()(e1)

    e2 = Conv1D(filters=filters, kernel_size=kernel_size+3, strides=strides+2, padding='same')(input_tensor)
    e2 = BatchNormalization()(e2)
    e2 = PReLU()(e2)

    con_e = concatenate([e2, e1], axis=-1)
    out_con = Conv1D(1, 1, activation='tanh')(con_e)
    out_con = Add()([out_con, input_tensor])
    # outputs = tf.nn.PReLU(out_con)

    return out_con