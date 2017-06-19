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
import os.path
import time
import logging
import json
import importlib

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import tensorflow.python.debug as tf_debug

PHASE = 'train'
# import json_maker, update json files and read requested json file
import Model_Settings.json_maker as json_maker
json_maker.recompile_json_files()
jsonToRead = '170523_ITR_B_1.json'
print("Reading %s" % jsonToRead)
with open('Model_Settings/'+jsonToRead) as data_file:
    modelParams = json.load(data_file)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# import input & output modules 
import Data_IO.data_input as data_input
import Data_IO.data_output as data_output

# import corresponding model name as model_cnn, specifed at json file
model_cnn = importlib.import_module('Model_Factory.'+modelParams['modelName'])

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
def _get_control_params():
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

def train():
    _get_control_params()

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
        images, pclA, pclB, targetT, tfrecFileIDs = data_input.inputs(**modelParams)
        print('Input        ready')
        # Build a Graph that computes the HAB predictions from the
        # inference model.
        targetP = model_cnn.inference(images, **modelParams)

        # Calculate loss. 2 options:

        # use mask to get degrees significant
        # What about adaptive mask to zoom into differences at each CNN stack !!!
        #loss = model_cnn.weighted_loss(targetP, targetT, **modelParams)
        loss = model_cnn.weighted_params_loss(targetP, targetT, **modelParams)
        # pcl based loss
        #loss = model_cnn.pcl_params_loss(pclA, targetP, targetT, **modelParams)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        opTrain = model_cnn.train(loss, globalStep, **modelParams)
        ##############################
        print('Training     ready')
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        print('Saver        ready')

        # Build the summary operation based on the TF collection of Summaries.
        summaryOp = tf.summary.merge_all()
        print('MergeSummary ready')

        # Build an initialization operation to run below.
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        opCheck = tf.add_check_numerics_ops()
        # Start running operations on the Graph.
        config = tf.ConfigProto(log_device_placement=modelParams['logDevicePlacement'])
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.Session(config=config)
        print('Session      ready')


        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        print('QueueRunner  started')

        summaryWriter = tf.summary.FileWriter(modelParams['trainLogDir'], sess.graph)
        
        print('Training     started')
        durationSum = 0
        durationSumAll = 0
        for step in range(1):# xrange(modelParams['maxSteps']):
            startTime = time.time()
            _, lossValue = sess.run([opTrain, loss])
            duration = time.time() - startTime
            durationSum += duration
            assert not np.isnan(lossValue), 'Model diverged with loss = NaN'

            if step % FLAGS.printOutStep == 0:
                numExamplesPerStep = modelParams['activeBatchSize']
                examplesPerSec = numExamplesPerStep / duration
                secPerBatch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch), loss/batch = %.2f')
                logging.info(format_str % (datetime.now(), step, lossValue,
                                           examplesPerSec, secPerBatch, lossValue/modelParams['activeBatchSize']))

            if step % FLAGS.summaryWriteStep == 0:
                summaryStr = sess.run(summaryOp)
                summaryWriter.add_summary(summaryStr, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.modelCheckpointStep == 0 or (step + 1) == modelParams['maxSteps']:
                checkpointPath = os.path.join(modelParams['trainLogDir'], 'model.ckpt')
                saver.save(sess, checkpointPath, global_step=step)

            # Print Progress Info
            if ((step % FLAGS.ProgressStepReportStep) == 0) or ((step+1) == modelParams['maxSteps']):
                print('Progress: %.2f%%, Elapsed: %.2f mins, Training Completion in: %.2f mins' %
                        ((100*step)/modelParams['maxSteps'], durationSum/60, (((durationSum*modelParams['maxSteps'])/(step+1))/60)-(durationSum/60)))
                
        ######### USE LATEST STATE TO WARP IMAGES
        durationSum = 0
        durationSumAll = 0
        if modelParams['writeWarpedImages']:
            lossValueSum = 0
            stepsForOneDataRound = int((modelParams['numExamples']/modelParams['activeBatchSize']))+1
            print('Warping images with batch size %d in %d steps' % (modelParams['activeBatchSize'], stepsForOneDataRound))
            for step in xrange(stepsForOneDataRound):
                startTime = time.time()
                evImages, evPclA, evPclB, evtargetT, evtargetP, evtfrecFileIDs, evlossValue = sess.run([images, pclA, pclB, targetT, targetP, tfrecFileIDs, loss])
                duration = time.time() - startTime
                durationSum += duration
                #### put imageA, warpped imageB by pHAB, HAB-pHAB as new HAB, changed fileaddress tfrecFileIDs
                data_output.output(evImages, evPclA, evPclB, evtargetT, evtargetP, evtfrecFileIDs, **modelParams)
                duration = time.time() - startTime
                durationSumAll += duration
                # Print Progress Info
                if ((step % FLAGS.ProgressStepReportOutputWrite) == 0) or ((step+1) == stepsForOneDataRound):
                    print('Progress: %.2f%%, Loss: %.2f, Elapsed: %.2f mins, Training Completion in: %.2f mins' % 
                            ((100*step)/stepsForOneDataRound, evlossValue/(step+1), durationSum/60, (((durationSum*stepsForOneDataRound)/(step+1))/60)-(durationSum/60)))
                    #print('Total Elapsed: %.2f mins, Training Completion in: %.2f mins' % 
                    #        durationSumAll/60, (((durationSumAll*stepsForOneDataRound)/(step+1))/60)-(durationSumAll/60))
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
    if input("(Overwrite WARNING) Did you change logs directory? ") != "yes":
        print("Please consider changing logs directory in order to avoid overwrite!")
        return
    if tf.gfile.Exists(modelParams['trainLogDir']):
        tf.gfile.DeleteRecursively(modelParams['trainLogDir'])
    tf.gfile.MakeDirs(modelParams['trainLogDir'])
    train()


if __name__ == '__main__':
    # looks up in the module named "__main__" (which is this module since its whats being run) in the sys.modules
    # list and invokes that modules main function (defined above)
    #    - in the process it parses the command line arguments using tensorflow.python.platform.flags.FLAGS
    #    - run can be called with the specific main and/or arguments
    tf.app.run()
