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
from __future__ import absolute_import
from __future__ import division

from datetime import datetime
import os, os.path
import time
import logging
import json
import importlib

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import Data_IO.data_input_ntuple as data_input
import Data_IO.data_output_ntuple as data_output

PHASE = 'train'

####################################################
####################################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('printOutStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('summaryWriteStep', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('modelCheckpointStep', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportStep', 250,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('ProgressStepReportOutputWrite', 250,
                            """Number of batches to run.""")
####################################################
####################################################
def _set_control_params(modelParams):
    modelParams['phase'] = PHASE
    #params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])

    modelParams['existingParams'] = None

    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTrainDataDir']

    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTestDataDir']
    return modelParams
####################################################
####################################################
################### introducing new op for mod
def np_mod(x,y):
    return (x % y).astype(np.float32)

def modgrad(op, grad):
    x = op.inputs[0] # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )
    y = op.inputs[1] # the second argument
    return grad * 1, grad * tf.negative(tf.floordiv(x, y)) #the propagated gradient with respect to the first and second argument respectively

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

from tensorflow.python.framework import ops
def tf_mod(x,y, name=None):

    with ops.op_scope([x,y], name, "mod") as name:
        z = py_func(np_mod,
                        [x,y],
                        [tf.float32],
                        name=name,
                        grad=modgrad)  # <-- here's the call to the gradient
        return z[0]

@tf.RegisterGradient("mod")
def _sub_grad(unused_op, grad):
  return grad, tf.negative(grad)
###################
def pcl_loss(pclA, tMatP, tMatT, **kwargs): # batchSize=Sne
    """
    Generate a ground truth point cloud using ground truth transformation
    Generate a prediction point cloud using predicted transformation
    L2 difference between ground truth and predicted point cloud is the loss value
    """
    # pclA, tMatP, tMatT are in batches
    ## # tMatP, tMatT should get a 0,0,0,1 row and be reshaped to 4x4
    ## tMatP = tf.concat([tMatP, tf.constant(np.repeat(np.array([[0, 0, 0, 1]],
    ##                                                          dtype=np.float32),
    ##                                                 kwargs.get('activeBatchSize'),
    ##                                                 axis=0))],
    ##                   1)
    ## tMatT = tf.concat([tMatT, tf.constant(np.repeat(np.array([[0, 0, 0, 1]],
    ##                                                          dtype=np.float32),
    ##                                                 kwargs.get('activeBatchSize'),
    ##                                                 axis=0))],
    ##                   1)
    tMatP = tf.reshape(tMatP, [kwargs.get('activeBatchSize'), 4, 4])
    tMatT = tf.reshape(tMatT, [kwargs.get('activeBatchSize'), 4, 4])
    # pclA should get a row of ones
    pclA = tf.concat([pclA, tf.constant(np.ones([kwargs.get('activeBatchSize'), 1, kwargs.get('pclCols')],
                                                dtype=np.float32))],
                     1)
    pclP = tf.matmul(tMatP, pclA)
    pclT = tf.matmul(tMatT, pclA)
    return model_cnn.loss(pclP, pclT, **kwargs)

def pcl_params_loss(pclA, pred, target, **kwargs): # batchSize=Sne
    """
    Generate transformation matrix using parameters for both prediction and ground truth
    Generate a ground truth point cloud using ground truth transformation
    Generate a prediction point cloud using predicted transformation
    L2 difference between ground truth and predicted point cloud is the loss value
    """
    print('paramsLoss')
    # pclA, tMatP, tMatT are in batches
    # tMatP, tMatT should get a 0,0,0,1 row and be reshaped to 4x4
    # transpose to easily extract columns: batchSize x 6 -> 6 x batchSize
    pred = tf.transpose(pred)
    target = tf.transpose(target)
    # generate tMatP and tMatT: 12 x batchSize
    a = pred[0]
    b = pred[1]
    g = pred[2]
    dx = pred[3]
    dy = pred[4]
    dz = pred[5]
    tMatP = tf.get_variable('tMatP', shape=[16,modelParams['activeBatchSize']], initializer=tf.constant_initializer(0.0), dtype=tf.float32, trainable=False)
    tMatP = tf.Variable([
              tf.cos(pred[0])*tf.cos(pred[1]), (tf.cos(pred[0])*tf.sin(pred[1])*tf.sin(pred[2]))-(tf.sin(pred[0])*tf.cos(pred[2])), (tf.cos(pred[0])*tf.sin(pred[1])*tf.cos(pred[2]))+(tf.sin(pred[0])*tf.sin(pred[2])), pred[3],
              tf.sin(pred[0])*tf.cos(pred[1]), (tf.sin(pred[0])*tf.sin(pred[1])*tf.sin(pred[2]))+(tf.cos(pred[0])*tf.cos(pred[2])), (tf.sin(pred[0])*tf.sin(pred[1])*tf.cos(pred[2]))-(tf.cos(pred[0])*tf.sin(pred[2])), pred[4],
              -tf.sin(pred[1]),                tf.cos(pred[1])*tf.sin(pred[2]),                                                     tf.cos(pred[1])*tf.cos(pred[2]),                                                     pred[5],
              pred[0]*0,                 pred[0]*0,                                                   pred[0]*0,                                                 pred[0]*0+1
            ], dtype=tf.float32, name='tMatP', trainable=False)
    a = target[0]
    b = target[1]
    g = target[2]
    dx = target[3]
    dy = target[4]
    dz = target[5]
    tMatT = tf.get_variable('tMatT', shape=[16,modelParams['activeBatchSize']], initializer=tf.constant_initializer(0.0), dtype=tf.float32, trainable=False)
    tMatT = tf.Variable([
              tf.cos(target[0])*tf.cos(target[1]), (tf.cos(target[0])*tf.sin(target[1])*tf.sin(target[2]))-(tf.sin(target[0])*tf.cos(target[2])), (tf.cos(target[0])*tf.sin(target[1])*tf.cos(target[2]))+(tf.sin(target[0])*tf.sin(target[2])), target[3],
              tf.sin(target[0])*tf.cos(target[1]), (tf.sin(target[0])*tf.sin(target[1])*tf.sin(target[2]))+(tf.cos(target[0])*tf.cos(target[2])), (tf.sin(target[0])*tf.sin(target[1])*tf.cos(target[2]))-(tf.cos(target[0])*tf.sin(target[2])), target[4],
              -tf.sin(target[1]),          tf.cos(pred[1])*tf.sin(target[2]),                                   tf.cos(target[1])*tf.cos(target[2]),                                   target[5],
              target[0]*0,                 target[0]*0,                                                   target[0]*0,                                                 target[0]*0+1
           ], dtype=tf.float32, name='tMatT', trainable=False)
    # convert tMat's to correct form: 12 x batchSize -> batchSize x 12
    tMatP = tf.transpose(tMatP)
    tMatT = tf.transpose(tMatT)
    print('paramsLossDone')

    return pcl_loss(pclA, tMatP, tMatT, **kwargs)
####################################################
####################################################
def train(modelParams):
    # import corresponding model name as model_cnn, specifed at json file
    model_cnn = importlib.import_module('Model_Factory.'+modelParams['modelName'])

    if not os.path.exists(modelParams['dataDir']):
        raise ValueError("No such data directory %s" % modelParams['dataDir'])

    _setupLogging(os.path.join(modelParams['trainLogDir'], "genlog"))

    with tf.Graph().as_default():
        # track the number of train calls (basically number of batches processed)
        globalStep = tf.get_variable('globalStep',
                                     [],
                                     initializer=tf.constant_initializer(0),
                                     trainable=False)

        # Get images and transformation for model_cnn.
        images, pclA, targetT, tfrecFileIDs = data_input.inputs(**modelParams)
        print('Input        ready')
        # Build a Graph that computes the HAB predictions from the
        # inference model.
        targetP = model_cnn.inference(images, **modelParams)

        # Calculate loss. 2 options:

        # use mask to get degrees significant
        # What about adaptive mask to zoom into differences at each CNN stack !!!
        ########## model_cnn.loss is called in the loss function
        #loss = weighted_loss(targetP, targetT, **modelParams)       
        if True:#modelParams.get('lastTuple'):
            # Training with all, loss on the last tuple only
            print("Using all ", modelParams.get('numTuple'), " to predict (rgs) only the LAST transformation")
            # loss on last tuple
            loss = model_cnn.loss(targetP, targetT[:,:,modelParams['numTuple']-2:modelParams['numTuple']-1], **modelParams)
        else:
            # Training on all, loss on all
            print("Using all ", modelParams.get('numTuple'), " to predict (rgs) all", modelParams.get('numTuple'), " transformations")
            # for training on all tuples
            loss = model_cnn.loss(targetP, targetT, **modelParams)
        
        # modelParams[imageDepthChannels]-2 cuz we have n tuples and n-1 transitions => changing index to 0 means depth-2
        #loss = model_cnn.loss(targetP, targetT[:,:,modelParams['imageDepthChannels']-2], **modelParams)
        # in a linear target last output size is the last output
        #loss = model_cnn.loss(targetP, targetT[:,(modelParams['imageDepthChannels']-2)*modelParams['outputSize']:(modelParams['imageDepthChannels']-1)*modelParams['outputSize']], **modelParams)

        # pcl based loss
        #loss = pcl_params_loss(pclA, targetP, targetT, **modelParams)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        print('Saver        ready')

        # Build the summary operation based on the TF collection of Summaries.
        summaryOp = tf.summary.merge_all()
        print('MergeSummary ready')

        # Build an initialization operation to run below.
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        #opCheck = tf.add_check_numerics_ops()
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=modelParams['logDevicePlacement'])
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        print('Session      ready')

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)


        # restore a saver.

        saver.restore(sess, (modelParams['trainLogDir']+'/model.ckpt-'+str(modelParams['trainMaxSteps']-1)))
        print('     loading -> ', modelParams['trainLogDir']+'/model.ckpt-'+str(modelParams['trainMaxSteps']-1))
        print('Ex-Model     loaded')

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        print('QueueRunner  started')

        summaryWriter = tf.summary.FileWriter(modelParams['trainLogDir'], sess.graph)
        
        print('Training     started')
        ######### USE LATEST STATE TO WARP IMAGES
        filesDictionaryAccum = {}
        durationSum = 0
        durationSumAll = 0
        if modelParams['writeWarpedImages']:
            outputDIR = modelParams['warpedOutputFolder']+'/'
            print("Using final training state to output processed tfrecords\noutput folder: ", outputDIR)
            if tf.gfile.Exists(outputDIR):
                tf.gfile.DeleteRecursively(outputDIR)
            tf.gfile.MakeDirs(outputDIR)
            
            lossValueSum = 0
            stepsForOneDataRound = int((modelParams['numExamples']/modelParams['activeBatchSize']))+1
            print('Warping %d images with batch size %d in %d steps' % (modelParams['numExamples'], modelParams['activeBatchSize'], stepsForOneDataRound))
            for step in xrange(stepsForOneDataRound):
                startTime = time.time()
                evImages, evPcl, evtargetT, evtargetP, evtfrecFileIDs, evlossValue = sess.run([images, pclA, targetT, targetP, tfrecFileIDs, loss])
                for fileIdx in range(modelParams['activeBatchSize']):
                    fileIDname = str(evtfrecFileIDs[fileIdx][0]) + "_" + str(evtfrecFileIDs[fileIdx][1]) + "_" + str(evtfrecFileIDs[fileIdx][2])
                    if (fileIDname in filesDictionaryAccum):
                        filesDictionaryAccum[fileIDname]+=1
                    else:
                        filesDictionaryAccum[fileIDname]=1
                #### put imageA, warpped imageB by pHAB, HAB-pHAB as new HAB, changed fileaddress tfrecFileIDs
                data_output.output(evImages, evPcl, evtargetT, evtargetP, evtfrecFileIDs, **modelParams)
                duration = time.time() - startTime
                durationSum += duration
                durationSumAll += duration
                # Print Progress Info
                if ((step % FLAGS.ProgressStepReportStep) == 0) or ((step+1) == stepsForOneDataRound):
                    print('Number of files used in training', len(filesDictionaryAccum))
                    print('Progress: %.2f%%, Loss: %.2f, Elapsed: %.2f mins, Training Completion in: %.2f mins --- %s' % 
                            (
                                (100*step)/stepsForOneDataRound,
                                evlossValue/(step+1), durationSum/60,
                                (((durationSum*stepsForOneDataRound)/(step+1))/60)-(durationSum/60),
                                datetime.now()
                            )
                        )
                    #print('Total Elapsed: %.2f mins, Total Completion in: %.2f mins' % (durationSumAll/60), ((((durationSumAll*stepsForOneDataRound)/(step+1))/60)-(durationSumAll/60)) )
            print('Number of files used in training', len(filesDictionaryAccum))
            filesAccum = np.array(list(filesDictionaryAccum.values()))
            print('Access statistics for each file, mean max min std', np.mean(filesAccum), np.max(filesAccum), np.min(filesAccum), np.std(filesAccum))
            print('Average training loss = %.2f - Average time per sample= %.2f s, Steps = %d' % (evlossValue/modelParams['activeBatchSize'], durationSum/(step*modelParams['activeBatchSize']), step))



def _setupLogging(logPath):
    # cleanup
    if os.path.isfile(logPath):
        os.remove(logPath)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logPath,
                        filemode='w')

    # also write out to the console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))

    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    logging.info("Logging setup complete to %s" % logPath)

def main(argv=None):  # pylint: disable=unused-argumDt
    if (len(argv)<3):
        print("Enter 'model name' and 'iteration number'")
        return
    modelName = argv[1]
    itrNum = int(argv[2])
    if itrNum>4 or itrNum<0:
        print('iteration number should only be from 1 to 4 inclusive')
        return
    # import json_maker, update json files and read requested json file
    import Model_Settings.json_maker as json_maker
    if not json_maker.recompile_json_files(modelName, itrNum):
        return
    jsonToRead = modelName+'_'+str(itrNum)+'.json'
    print("Reading %s" % jsonToRead)
    with open('Model_Settings/'+jsonToRead) as data_file:
        modelParams = json.load(data_file)

    modelParams = _set_control_params(modelParams)

    print(modelParams['modelName'])
    print('Rounds on datase = %.1f' % float((modelParams['trainBatchSize']*modelParams['trainMaxSteps'])/modelParams['numTrainDatasetExamples']))
    print('Train Input: %s' % modelParams['trainDataDir'])
    #print('Test  Input: %s' % modelParams['testDataDir'])
    print('Train Logs Output: %s' % modelParams['trainLogDir'])
    #print('Test  Logs Output: %s' % modelParams['testLogDir'])
    print('Train Warp Output: %s' % modelParams['warpedTrainDataDir'])
    #print('Test  Warp Output: %s' % modelParams['warpedTestDataDir'])
    print('')
    print('')
    #if input("(Overwrite WARNING) Did you change logs directory? (y) ") != "y":
    #    print("Please consider changing logs directory in order to avoid overwrite!")
    #    return
    #if tf.gfile.Exists(modelParams['trainLogDir']):
    #    tf.gfile.DeleteRecursively(modelParams['trainLogDir'])
    #tf.gfile.MakeDirs(modelParams['trainLogDir'])
    train(modelParams)



if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()