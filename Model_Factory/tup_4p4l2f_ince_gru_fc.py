# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the calusa_heatmap network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

import Model_Factory.model_base as model_base

USE_FP_16 = False

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def inference(images, **kwargs): #batchSize=None, phase='train', outLayer=[13,13], existingParams=[]
    modelShape = kwargs.get('modelShape')
    numParalModules = kwargs.get('numParallelModules')
    wd = None #0.0002
    USE_FP_16 = kwargs.get('usefp16')
    dtype = tf.float16 if USE_FP_16 else tf.float32

    batchSize = kwargs.get('activeBatchSize', None)

    ############# CONV1_TWIN 3x3 conv, 2 input dims, 2 parallel modules, 64 output dims (filters)
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv11', images, kwargs.get('imageDepthChannels'), numParalModules,
                                                                  {'cnn1x1': modelShape[0], 'cnn3x3': modelShape[0], 'cnn5x5': modelShape[0]},
                                                                  wd, **kwargs)
    # calc batch norm CONV1_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV2_TWIN    
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv12', fireOut, prevExpandDim, numParalModules,
                                                                  {'cnn1x1': modelShape[1]},
                                                                  wd, **kwargs)
    # calc batch norm CONV2_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # Pooling1 2x2 wit stride 2
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],
                          padding='SAME', name='maxpool1')
    ############# CONV3_TWIN
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv21', fireOut, prevExpandDim, numParalModules,
                                                                  {'cnn1x1': modelShape[2], 'cnn3x3': modelShape[2]},
                                                                  wd, **kwargs)
    # calc batch norm CONV3_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV3_TWIN
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv22', fireOut, prevExpandDim, numParalModules,
                                                                  {'cnn1x1': modelShape[3]},
                                                                  wd, **kwargs)
    # calc batch norm CONV3_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # Pooling1 2x2 wit stride 2
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool2')
    ############# CONV4_TWIN
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv31', fireOut, prevExpandDim, numParalModules,
                                                                  {'cnn1x1': modelShape[4], 'cnn3x3': modelShape[4]},
                                                                  wd, **kwargs)
    # calc batch norm CONV4_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # Pooling2 2x2 wit stride 2
    ############# CONV4_TWIN
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv32', fireOut, prevExpandDim, numParalModules,
                                                                  {'cnn1x1': modelShape[5]},
                                                                  wd, **kwargs)
    # calc batch norm CONV4_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # Pooling2 2x2 wit stride 2
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool3')
    ############# CONV4_TWIN
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv41', fireOut, prevExpandDim, numParalModules,
                                                                  {'cnn1x1': modelShape[6], 'cnn3x3': modelShape[6]},
                                                                  wd, **kwargs)
    # calc batch norm CONV4_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # Pooling2 2x2 wit stride 2
    ############# CONV4_TWIN
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv42', fireOut, prevExpandDim, numParalModules,
                                                                  {'cnn1x1': modelShape[7]},
                                                                  wd, **kwargs)
    # calc batch norm CONV4_TWIN
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # Pooling2 2x2 wit stride 2
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool4')
    ################################# Correlate Sequential Data to each other t, t+1
    ## We have data as [B, r, c, nt*d]. We transform it to [B, r, c, (nt-1)*2d]
    fireOut = tf.split(fireOut, numParalModules, 3) # split along last dimension to [nt] places
    fireOut[0] = tf.concat([fireOut[0], fireOut[1]], 3)
    numSeqModules = numParalModules-1
    for spl in range(1, numSeqModules):
        fireOut[0] = tf.concat([fireOut[0], fireOut[spl], fireOut[spl+1]], 3)
    fireOut = fireOut[0]
    prevExpandDim = int(fireOut.get_shape()[3])
    print('+++++ in_seq', fireOut.get_shape())
    ############# CONV5
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv51', fireOut, prevExpandDim, numSeqModules,
                                                                  {'cnn1x1': modelShape[8], 'cnn3x3': modelShape[8]},
                                                         wd, **kwargs)
    # calc batch norm CONV5
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV5
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv52', fireOut, prevExpandDim, numSeqModules,
                                                                  {'cnn1x1': modelShape[9]},
                                                         wd, **kwargs)
    # calc batch norm CONV5
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # Pooling2 2x2 wit stride 2
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool5')
    ############# CONV6
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv61', fireOut, prevExpandDim, numSeqModules,
                                                                  {'cnn1x1': modelShape[10], 'cnn3x3': modelShape[10]},
                                                         wd, **kwargs)
    # calc batch norm CONV6
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV6
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv62', fireOut, prevExpandDim, numSeqModules,
                                                                  {'cnn1x1': modelShape[11]},
                                                         wd, **kwargs)
    # calc batch norm CONV6
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    # Pooling2 2x2 wit stride 2
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool6')
    ############# CONV7
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv71', fireOut, prevExpandDim, numSeqModules,
                                                                  {'cnn1x1': modelShape[12], 'cnn3x3': modelShape[12]},
                                                         wd, **kwargs)
    # calc batch norm CONV7
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV7
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv72', fireOut, prevExpandDim, numSeqModules,
                                                                  {'cnn1x1': modelShape[13]},
                                                         wd, **kwargs)
    # calc batch norm CONV7
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ###### Pooling2 2x2 wit stride 2
    fireOut = tf.nn.max_pool(fireOut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='maxpool7')
    ############# CONV8
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv81', fireOut, prevExpandDim, numSeqModules,
                                                                  {'cnn1x1': modelShape[14], 'cnn3x3': modelShape[14]},
                                                         wd, **kwargs)
    # calc batch norm CONV8
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# CONV8
    fireOut, prevExpandDim = model_base.conv_fire_parallel_inception_module('conv82', fireOut, prevExpandDim, numSeqModules,
                                                                  {'cnn1x1': modelShape[15]},
                                                         wd, **kwargs)
    # calc batch norm CONV8
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ###### DROPOUT after CONV8
    with tf.name_scope("drop"):
        keepProb = tf.constant(kwargs.get('dropOutKeepRate') if kwargs.get('phase') == 'train' else 1.0, dtype=dtype)
        fireOut = tf.nn.dropout(fireOut, keepProb, name="dropout")
    
    ###### Prepare for fully connected layers
    print('+++++ FC_inp', fireOut.get_shape(), numSeqModules)
    # Reshape firout for LSTM
    # fireout =  [B, r, c, (nt-1)*d]
    #  split---> (nt-1)[B, r, c, d]
    #  stack---> [nt-1, B, r, c, d] ## Static_RNN uses this format
    #  swap ---> [B, nt-1, r, c, d] ## Dynamic_RNN uses this format -- BETTER 
    # reshape = [B, nt-1, r*c*d]
    ### RNN ---> time_major = False
    fireOut = tf.transpose(tf.stack(tf.split(fireOut, numSeqModules, 3), 0), perm=[1,0,2,3,4])
    fireOut = tf.reshape(fireOut, [batchSize, numSeqModules, -1])
    ### RNN ---> time_major = true
    #fireOut = tf.stack(tf.split(fireOut, numSeqModules, 3), 0)
    #fireOut = tf.reshape(fireOut, [numSeqModules, batchSize, -1])
    print('+++++ de_seq', fireOut.get_shape())
    prevExpandDim = int(fireOut.get_shape()[2])
    ############# FC1-LSTM layer with 1024 hidden celss
    fireOut, prevExpandDim = model_base.fc_fire_GRU_module('fcgru1', fireOut, prevExpandDim,
                                                       {'fcgru': modelShape[16]},
                                                       wd, **kwargs)
    # calc batch norm FC1
    if kwargs.get('batchNorm'):
        fireOut = model_base.batch_norm('batch_norm', fireOut, dtype)
    ############# FC2 layer with 8 outputs
    fireOut, prevExpandDim = model_base.fc_regression_module('fc2', fireOut, prevExpandDim,
                                                             {'fc': kwargs.get('outputSize')},
                                                             wd, **kwargs)
    return fireOut

def loss(pred, target, **kwargs): # batchSize=Sne
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size, heatmap_size ]
    Returns:
      Loss tensor of type float.
    """
    return model_base.loss(pred, target, **kwargs)

def train(loss, globalStep, **kwargs):
    return model_base.train(loss, globalStep, **kwargs)

def test(loss, globalStep, **kwargs):
    return model_base.test(loss, globalStep, **kwargs)
