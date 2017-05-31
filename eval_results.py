from datetime import datetime
import os.path
import time
import json
import importlib
from os import listdir
from os.path import isfile, join

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import Data_IO.kitti_shared as kitti

# import json_maker, update json files and read requested json file
import Model_Settings.json_maker as json_maker
json_maker.recompile_json_files()
jsonToRead = '170523_ITR_B_1.json'
print("Reading %s" % jsonToRead)
with open('Model_Settings/'+jsonToRead) as data_file:
    modelParams = json.load(data_file)

############# STATE
PHASE = 'train' # 'train' or 'test'
############# PATHS
pclPath = '../Data/kitti/pointcloud/'
posePath = '../Data/kitti/poses/'
seqIDtrain = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
seqIDtest = ['09', '10']
####################################################
def _get_file_names(readFolder, fileFormat):
    print(readFolder)
    print(fileFormat)
    filenames = [f for f in listdir(readFolder) if (isfile(join(readFolder, f)) and fileFormat in f)]
    filenames.sort()
    return filenames

def _get_prediction(pFilenames, seqID):
    
    newlist = sorted(list_to_be_sorted, key=lambda k: k['name']) 
    for i in range(len(pFilenames))
    with open('Model_Settings/'+jsonToRead) as data_file:
    modelParams = json.load(data_file)

def _get_control_params():
    modelParams['phase'] = PHASE
    #params['shardMeta'] = model_cnn.getShardsMetaInfo(FLAGS.dataDir, params['phase'])

    modelParams['existingParams'] = None
    modelParams['gTruthDir'] = posePath

    if modelParams['phase'] == 'train':
        modelParams['activeBatchSize'] = modelParams['trainBatchSize']
        modelParams['maxSteps'] = modelParams['trainMaxSteps']
        modelParams['numExamples'] = modelParams['numTrainDatasetExamples']
        modelParams['dataDir'] = modelParams['trainDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTrainDataDir']
        modelParams['tMatDir'] = modelParams['tMatTrainDir']
        modelParams['seqIDs'] = seqIDtrain

    if modelParams['phase'] == 'test':
        modelParams['activeBatchSize'] = modelParams['testBatchSize']
        modelParams['maxSteps'] = modelParams['testMaxSteps']
        modelParams['numExamples'] = modelParams['numTestDatasetExamples']
        modelParams['dataDir'] = modelParams['testDataDir']
        modelParams['warpedOutputFolder'] = modelParams['warpedTestDataDir']
        modelParams['tMatDir'] = modelParams['tMatTestDir']
        modelParams['seqIDs'] = seqIDtest

def evaluate():
    # Read all prediction posefiles and sort them based on the seqID and frameID
    pFilenames = _get_file_names(modelParams['tMatDir'])

    # For each sequence
    for i in range(len(modelParams['seqIDs'])):
        print('Processing sequences: %d / %d' % i, len(modelParams['seqIDs']))
        # Read groundtruth posefiles
        posePath = kitti.get_pose_path(modelParams['gTruthDir'], modelParams['seqIDs'][i])
        poseFile = kitti.get_pose_data(posePath)
    #  

def main(argv=None):  # pylint: disable=unused-argumDt
    _get_control_params()
    print('Evaluation phase for %s' % modelParams['phase'])
    print('Ground truth input: %s' % modelParams['gTruthDir'])
    if modelParams['phase'] == 'train':
        print('Train sequences:', seqIDtrain)
        print('Prediction input: %s' % modelParams['tMatDir'])
    else:
        print('Test sequences:' % seqIDtest)
        print('Prediction Input: %s' % modelParams['tMatDir'])
    print(modelParams['modelName'])
    pFilenames = _get_file_names(modelParams['tMatDir'], "")
    print(len(pFilenames))
    print(pFilenames[0:35])
    if input("IS PRESENTED INFORMATION VALID? ") != "yes":
        print("Please consider updating the provided information!")
        return
    evaluate()

main()