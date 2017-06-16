import json
import collections
import numpy as np
import os

def write_json_file(filename, datafile):
    filename = 'Model_Settings/'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent=0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
# Twin Common Parameters
trainLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/train_logs/'
testLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/test_logs/'
warpedTrainDirBase = '../Data/kitti/train_tfrecords_iterative/'
warpedTestDirBase = '../Data/kitti/test_tfrecords_iterative/'
####################################################################################
# Data Parameters
numTrainDatasetExamples_desc = "Number of images to process in train dataset"
numTestDatasetExamples_desc = "Number of images to process in test dataset"
trainDataDir_desc = "Directory to read training samples"
testDataDir_desc = "Directory to read test samples"
warpedTrainDataDir_desc = "Directory where to write wrapped train images"
warpedTestDataDir_desc = "Directory where to write wrapped test images"
trainLogDir_desc = "Directory where to write train event logs and checkpoints"
testLogDir_desc = "Directory where to write test event logs and checkpoints"
tMatTrainDir_desc = "tMat Output folder for train"
tMatTestDir_desc = "tMat Output folder for test"
writeWarped_desc = "Flag showing if warped images should be written"
pretrainedModelCheckpointPath_desc = "If specified, restore this pretrained model before beginning any training"
# Image Parameters
imageDepthRows_desc = "Depth Image Height (ROWS)"
imageDepthCols_desc = "Depth Image Width (COLS)"
imageDepthChannels_desc = "Depth Image channels, number of stacked images"
# PCL Parameters
pclRows_desc = "3 rows, xyz of Point Cloud"
pclCols_desc = "Unified number of points (columns) in the Point Cloud"
# tMat Parameters
tMatRows_desc = "rows in transformation matrix"
tMatCols_desc = "cols in transformation matrix"
# Model Parameters
modelName_desc = "Name of the model file to be loaded from Model_Factory"
modelShape_desc = "Network model with 8 convolutional layers with 2 fully connected layers"
numParallelModules_desc = "Number of parallel modules of the network"
batchNorm_desc = "Should we use batch normalization"
weightNorm_desc = "Should we use weight normalization"
optimizer_desc = "Type of optimizer to be used [AdamOptimizer, MomentumOptimizer, GradientDescentOptimizer]"
initialLearningRate_desc = "Initial learning rate."
learningRateDecayFactor_desc = "Learning rate decay factor"
numEpochsPerDecay_desc = "Epochs after which learning rate decays"
momentum_desc = "Momentum Optimizer: momentum"
epsilon_desc = "epsilon value used in AdamOptimizer"
dropOutKeepRate_desc = "Keep rate for drop out"
clipNorm_desc = "Gradient global normalization clip value"
lossFunction_desc = "Indicates type of the loss function to be used [L2, CrossEntropy, ..]"
# Train Parameters
trainBatchSize_desc = "Batch size of input data for train"
testBatchSize_desc = "Batch size of input data for test"
outputSize_desc = "Final output size"
trainMaxSteps_desc = "Number of batches to run"
testMaxSteps_desc = "Number of batches to run during test. numTestDatasetExamples = testMaxSteps x testBatchSize" 
usefp16_desc = "Use 16 bit floating point precision"
logDevicePlacement_desc = "Whether to log device placement"

data = {
    # Data Parameters
    'numTrainDatasetExamples' : 20400,
    'numTestDatasetExamples' : 2790,
    'trainDataDir' : '../Data/kitti/train_tfrecords',
    'testDataDir' : '../Data/kitti/test_tfrecords',
    'warpedTrainDataDir' : warpedTrainDirBase+'',
    'warpedTestDataDir' : warpedTestDirBase+'',
    'trainLogDir' : trainLogDirBase+'',
    'testLogDir' : testLogDirBase+'',
    'tMatTrainDir' : trainLogDirBase+'/target',
    'tMatTestDir' : testLogDirBase+'/target',
    'writeWarped' : False,
    'pretrainedModelCheckpointPath' : '',
    # Image Parameters
    'imageDepthRows' : 128,
    'imageDepthCols' : 512,
    'imageDepthChannels' : 2, # All PCL files should have same cols
    # PCL Parameters
    'pclRows' : 3,
    'pclCols' : 62074,
    # tMat Parameters
    'tMatRows' : 3,
    'tMatCols' : 4,
    # Model Parameters
    'modelName' : '',
    'modelShape' : [64, 64, 64, 64, 128, 128, 128, 128, 1024],
    'numParallelModules' : 2,
    'batchNorm' : True,
    'weightNorm' : False,
    'optimizer' : 'MomentumOptimizer', # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
    'initialLearningRate' : 0.005,
    'learningRateDecayFactor' : 0.1,
    'numEpochsPerDecay' : 10000.0,
    'momentum' : 0.9,
    'epsilon' : 0.1,
    'dropOutKeepRate' : 0.5,
    'clipNorm' : 1.0,
    'lossFunction' : 'L2',
    # Train Parameters
    'trainBatchSize' : 16,
    'testBatchSize' : 16,
    'outputSize' : 6, # 6 Params
    'trainMaxSteps' : 30000,
    'testMaxSteps' : 1,
    'usefp16' : False,
    'logDevicePlacement' : False
    }
data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
####################################################################################
####################################################################################
####################################################################################
####################################################################################
##############
reCompileJSON = True

############## TWIN
def write_twin():
    reCompileTwins = True
    NOrecompileTwins = False
    # Twin Common Parameters
    trainLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/train_logs/'
    testLogDirBase = '../Data/kitti/logs/tfdh_twin_py_logs/test_logs/'
    trainDataDir = '../Data/kitti/train_tfrecords'
    testDataDir = '../Data/kitti/test_tfrecords'

    data['modelName'] = 'twin_cnn_4p4l2f'

    writeWarpedImages = False

    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170127_TWN_MOM_W'
        data['testLogDir'] = testLogDirBase+'170127_TWN_MOM_W'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = False
        data['weightNorm'] = True
        write_json_file('170127_TWN_MOM_W.json', data)
    
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170127_TWN_MOM_B'
        data['testLogDir'] = testLogDirBase+'170127_TWN_MOM_B'
        data['trainMaxSteps'] = 120000
        data['numEpochsPerDecay'] = 40000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170127_TWN_MOM_B.json', data)
    
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170TWN_MOM_B'
        data['testLogDir'] = testLogDirBase+'170TWN_MOM_B'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 2048]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170TWN_MOM_B.json', data)
        
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170127_TWN_MOM_BW'
        data['testLogDir'] = testLogDirBase+'170127_TWN_MOM_BW'
        data['trainMaxSteps'] = 120000
        data['numEpochsPerDecay'] = 40000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = True
        write_json_file('170127_TWN_MOM_BW.json', data)
    ##############
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170118_AdamOpt_B16_256'
        data['testLogDir'] = testLogDirBase+'170118_AdamOpt_B16_256'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 16
        data['testBatchSize'] = 16
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 256, 256, 1024]
        data['batchNorm'] = False
        data['weightNorm'] = True
        write_json_file('170118_AdamOpt_B16_256.json', data)
    ###############
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170118_MomentumOpt_B20_256'
        data['testLogDir'] = testLogDirBase+'170118_MomentumOpt_B20_256'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 256, 256, 1024]
        data['batchNorm'] = False
        data['weightNorm'] = True
        write_json_file('170118_MomentumOpt_B20_256.json', data)
    ##############
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170120_MomentumOpt_256_256'
        data['testLogDir'] = testLogDirBase+'170120_MomentumOpt_256_256'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 256, 256, 256, 256, 1024]
        data['batchNorm'] = False
        data['weightNorm'] = True
        write_json_file('170120_MomentumOpt_256_256.json', data)
    
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170120_MomentumOpt_256_256_150k'
        data['testLogDir'] = testLogDirBase+'170120_MomentumOpt_256_256_150k'
        data['trainMaxSteps'] = 150000
        data['numEpochsPerDecay'] = 50000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 256, 256, 256, 256, 1024]
        data['batchNorm'] = False
        data['weightNorm'] = True
        write_json_file('170120_MomentumOpt_256_256_150k.json', data)
        
    ############## 
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170125_MomentumOpt_256_256_BNorm'
        data['testLogDir'] = testLogDirBase+'170125_MomentumOpt_256_256_BNorm'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 256, 256, 256, 256, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170125_MomentumOpt_256_256_BNorm.json', data)
    ##############
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170125_MomentumOpt_256_256_WBNorm'
        data['testLogDir'] = testLogDirBase+'170125_MomentumOpt_256_256_WBNorm'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 10
        data['testBatchSize'] = 10
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 256, 256, 256, 256, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = True
        write_json_file('170125_MomentumOpt_256_256_WBNorm.json', data)
    
    ##############
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170129_TWN_MOM_B_64'
        data['testLogDir'] = testLogDirBase+'170129_TWN_MOM_B_64'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170129_TWN_MOM_B_64.json', data)
    
    
    ##############
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170129_TWN_MOM_B_32'
        data['testLogDir'] = testLogDirBase+'170129_TWN_MOM_B_32'
        data['trainMaxSteps'] = 30000#90000
        data['numEpochsPerDecay'] = 10000.0#30000.0
        data['trainBatchSize'] = 60#20
        data['testBatchSize'] = 60#20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 32, 32, 32, 32, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170129_TWN_MOM_B_32.json', data)
        
    ##############
    if reCompileTwins:
        data['trainLogDir'] = trainLogDirBase+'170130_TWN_MOM_B_16'
        data['testLogDir'] = testLogDirBase+'170130_TWN_MOM_B_16'
        data['trainMaxSteps'] = 15000#90000
        data['numEpochsPerDecay'] = 5000.0#30000.0
        data['trainBatchSize'] = 120#20
        data['testBatchSize'] = 120#20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [16, 16, 16, 16, 16, 16, 16, 16, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170130_TWN_MOM_B_16.json', data)
    

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def write_single():
    # Single Common Parameters
    trainLogDirBase = '../Data/kitti/logs/tfdh_py_logs/train_logs/'
    testLogDirBase = '../Data/kitti/logs/tfdh_py_logs/test_logs/'
    data['trainDataDir'] = '../Data/kitti/train_tfrecords'
    data['testDataDir'] = '../Data/kitti/test_tfrecords'

    data['modelName'] = 'cnn_8l2f'

    writeWarpedImages = False

    ##############
    if reCompileJSON:
        data['trainLogDir'] = trainLogDirBase+'170126_SIN_B'
        data['testLogDir'] = testLogDirBase+'170126_SIN_B'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170126_SIN_B.json', data)

    if reCompileJSON:
        data['trainLogDir'] = trainLogDirBase+'170126_SIN_W'
        data['testLogDir'] = testLogDirBase+'170126_SIN_W'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = False
        data['weightNorm'] = True
        write_json_file('170126_SIN_W.json', data)

    if reCompileJSON:
        data['trainLogDir'] = trainLogDirBase+'170126_SIN_BW'
        data['testLogDir'] = testLogDirBase+'170126_SIN_BW'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = True
        write_json_file('170126_SIN_BW.json', data)
    ##############




##################################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def write_twin_correlation():
    # Twin Correlation Matching Common Parameters
    trainLogDirBase = '../Data/kitti/logs/tfdh_twincorr_logs/train_logs/'
    testLogDirBase = '../Data/kitti/logs/tfdh_twincorr_logs/test_logs/'
    trainDataDir = '../Data/kitti/train_tfrecords'
    testDataDir = '../Data/kitti/test_tfrecords'

    data['modelName'] = 'twin_cnn_4pCorr4l2f'

    writeWarpedImages = False

    ##############
    if reCompileJSON:
        data['trainLogDir'] = trainLogDirBase+'170131_TCOR_B'
        data['testLogDir'] = testLogDirBase+'170131_TCOR_B'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170131_TCOR_B.json', data)


####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def write_iterative():
    # Twin Correlation Matching Common Parameters
    trainLogDirBase = '../Data/kitti/logs/tfdh_iterative_logs/train_logs/'
    testLogDirBase = '../Data/kitti/logs/tfdh_iterative_logs/test_logs/'

    data['writeWarpedImages'] = True

    # Iterative model only changes the wayoutput is written, 
    # so any model can be used by ease

    reCompileITR = True
    NOreCompileITR = False
    ##############
    ##############
    ##############
    if reCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170523_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrecords'
        data['testDataDir'] = '../Data/kitti/test_tfrecords'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['warpOriginalImage'] = True
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170523_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170523_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170523_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['tMatTrainDir'] = data['trainLogDir']+'/target'
        data['tMatTestDir'] = data['testLogDir']+'/target'
        _set_folders(data['tMatTrainDir'])
        _set_folders(data['tMatTestDir'])
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
    ##############
    ##############
    ##############
    if NOreCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = 'GPUX_170301_ITR_B_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrecords'
        data['testDataDir'] = '../Data/kitti/test_tfrecords_ob_16'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = 'GPUX_170301_ITR_B_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = 'GPUX_170301_ITR_B_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = 'GPUX_170301_ITR_B_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

    ##############
    ##############
    ##############
    if NOreCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f'
        data['optimizer'] = 'AdamOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170301_ITR_B_ADAM_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrecords'
        data['testDataDir'] = '../Data/kitti/test_tfrecords'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170301_ITR_B_ADAM_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170301_ITR_B_ADAM_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170301_ITR_B_ADAM_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)


    ##############
    if NOreCompileITR:
        data['modelName'] = 'twin_cnn_4p4l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170301_ITR_B_512_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrecords'
        data['testDataDir'] = '../Data/kitti/test_tfrecords'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 512]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170301_ITR_B_512_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase + runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 512]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170301_ITR_B_512_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 512]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170301_ITR_B_512_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 512]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

    ##############
    if NOreCompileITR:
        data['modelName'] = 'twin_cnn_res_4p4l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170301_ITR_B_Res_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrecords'
        data['testDataDir'] = '../Data/kitti/test_tfrecords'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170301_ITR_B_Res_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170301_ITR_B_Res_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170301_ITR_B_Res_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        
    ##############
    if NOreCompileITR:
        data['modelName'] = 'twin_cnn_goog_res_4p4l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170301_ITR_B_GooGRes_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrecords'
        data['testDataDir'] = '../Data/kitti/test_tfrecords'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 2
        runName = '170301_ITR_B_GooGRes_2'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 3
        runName = '170301_ITR_B_GooGRes_3'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
        ### ITERATION 4
        runName = '170301_ITR_B_GooGRes_4'
        data['trainDataDir'] = data['warpedTrainDataDir'] # from previous iteration
        data['testDataDir'] = data['warpedTestDataDir'] # from previous iteration
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [32, 32, 32, 32, 64, 64, 64, 64, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)

    ##############
    if NOreCompileITR:
        data['modelName'] = 'twin_cnn_4p8l2f'
        data['optimizer'] = 'MomentumOptimizer' # AdamOptimizer MomentumOptimizer GradientDescentOptimizer
        ### ITERATION 1
        runName = '170311_ITR_B_12L_1'
        data['trainDataDir'] = '../Data/kitti/train_tfrecords'
        data['testDataDir'] = '../Data/kitti/test_tfrecords'
        data['trainLogDir'] = trainLogDirBase + runName
        data['testLogDir'] = testLogDirBase + runName
        data['warpedTrainDataDir'] = warpedTrainDirBase + runName
        data['warpedTestDataDir'] = warpedTestDirBase+ runName
        _set_folders(data['warpedTrainDataDir'])
        _set_folders(data['warpedTestDataDir'])
        data['warpOriginalImage'] = True
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 64
        data['testBatchSize'] = 64
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file(runName+'.json', data)
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################


def write_residual():
    # Twin Correlation Matching Common Parameters
    trainLogDirBase = '../Data/kitti/logs/tfdh_residual_logs/train_logs/'
    testLogDirBase = '../Data/kitti/logs/tfdh_residual_logs/test_logs/'

    data['modelName'] = 'twin_cnn_res_4p4l2f'

    writeWarpedImages = False
    
    ##############
    if reCompileJSON:
        data['trainLogDir'] = trainLogDirBase+'170213_TRES_B'
        data['testLogDir'] = testLogDirBase+'170213_TRES_B'
        data['trainMaxSteps'] = 90000
        data['numEpochsPerDecay'] = 30000.0
        data['trainBatchSize'] = 20
        data['testBatchSize'] = 20
        data['testMaxSteps'] = int(np.ceil(data['numTestDatasetExamples']/data['testBatchSize']))
        data['modelShape'] = [64, 64, 64, 64, 128, 128, 128, 128, 1024]
        data['batchNorm'] = True
        data['weightNorm'] = False
        write_json_file('170213_TRES_B.json', data)
        
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

def recompile_json_files():
    #write_single()
    #write_twin()
    #write_twin_correlation()
    write_iterative()
    #write_residual()
    print("JSON files updated")
