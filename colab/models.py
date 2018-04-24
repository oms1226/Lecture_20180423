import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from PIL import Image

import sys
sys.path.insert(0,'./')
from layers import *


def vgg_16(inputs):
    x = inputs
    with tf.variable_scope('conv1'):
        x = Conv2D(x, [3,3,3,64], [1,1,1,1], 'SAME', name='conv1_1')
        x = tf.nn.relu(x)
        x = Conv2D(x, [3,3,64,64], [1,1,1,1], 'SAME', name='conv1_2')
        x = tf.nn.relu(x)
    vgg12 = x  
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    with tf.variable_scope('conv2'):
        x = Conv2D(x, [3,3,64,128], [1,1,1,1], 'SAME', name='conv2_1')
        x = tf.nn.relu(x)
        x = Conv2D(x, [3,3,128,128], [1,1,1,1], 'SAME', name='conv2_2') 
        x = tf.nn.relu(x)
    vgg22 = x
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    with tf.variable_scope('conv3'):
        x = Conv2D(x, [3,3,128,256], [1,1,1,1], 'SAME', name='conv3_1')
        x = tf.nn.relu(x)
        x = Conv2D(x, [3,3,256,256], [1,1,1,1], 'SAME', name='conv3_2') 
        x = tf.nn.relu(x)        
        x = Conv2D(x, [3,3,256,256], [1,1,1,1], 'SAME', name='conv3_3')
        x = tf.nn.relu(x) 
    vgg33 = x            
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')    
        
    with tf.variable_scope('conv4'):
        x = Conv2D(x, [3,3,256,512], [1,1,1,1], 'SAME', name='conv4_1')
        x = tf.nn.relu(x)
        x = Conv2D(x, [3,3,512,512], [1,1,1,1], 'SAME', name='conv4_2') 
        x = tf.nn.relu(x)        
        x = Conv2D(x, [3,3,512,512], [1,1,1,1], 'SAME', name='conv4_3')
        x = tf.nn.relu(x)           
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID') 
    vgg43 = x
    with tf.variable_scope('conv5'):
        x = Conv2D(x, [3,3,512,512], [1,1,1,1], 'SAME', name='conv5_1')
        x = tf.nn.relu(x)
        x = Conv2D(x, [3,3,512,512], [1,1,1,1], 'SAME', name='conv5_2') 
        x = tf.nn.relu(x)        
        x = Conv2D(x, [3,3,512,512], [1,1,1,1], 'SAME', name='conv5_3')
        x = tf.nn.relu(x)
    vgg54 = x                
    
    return vgg12, vgg22, vgg33, vgg43, vgg54


def FCN(vgg54, batch_size):
    x = Conv2D(vgg54, [3,3,512,21], [1,1,1,1], 'SAME', name='new_conv1')
    x = DeConv2D(x, [32, 32, 21, 21], [batch_size, 128, 128, 21], [1, 16, 16, 1], 'SAME', name='new_deconv1')
    # x = tf.depth_to_space(x, 2)
    return x
    
def YOUR_MODEL(inputs, batch_size):
    x = inputs
    return x