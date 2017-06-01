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

def _get_all_predictions(pFilenames):
    """
    read all predictions of all sequences to a list
    """
    predAllList = list()
    seqList = list()
    with open(modelParams['tMatDir']+'/'+pFilenames[0]) as data_file:
        tMatJson = json.load(data_file)
    seqList.append(tMatJson)
    for i in range(1,len(pFilenames)):
        with open(modelParams['tMatDir']+'/'+pFilenames[i]) as data_file:
            tMatJson = json.load(data_file)
        condSameSeq = pFilenames[i][0] == pFilenames[i-1][0] and pFilenames[i][1] == pFilenames[i-1][1]
        if condSameSeq:
            seqList.append(tMatJson)
        if (not condSameSeq) or (i+1 == len(pFilenames)):
            seqList = sorted(seqList, key=lambda k: k['idx'])
            predAllList.append(seqList)
    return predAllList

def _get_prediction(predAllList, seqID):
    """
    get prediction for an specific sequence
    """
    return predAllList[int(seqID)]

def _get_tMat_A_2_B(tMatA2o, tMatB2o):
    '''
    tMatA2o A -> O (source pcl is in A), tMatB2o B -> O (target pcl will be in B)
    return tMat A -> B
    '''
    # tMatA2o: A -> Orig
    # tMatB2o: B -> Orig ==> inv(tMatB2o): Orig -> B
    # inv(tMatB2o) * tMatA2o : A -> B
    tMatA2o = np.append(tMatA2o, [[0, 0, 0, 1]], axis=0)
    tMatB2o = np.append(tMatB2o, [[0, 0, 0, 1]], axis=0)
    tMatA2B = np.matmul(np.linalg.inv(tMatB2o), tMatA2o)
    tMatA2B = np.delete(tMatA2B, tMatA2B.shape[0]-1, 0)
    return tMatA2B

def _get_gt_map_seq(gtPose):
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    pathMap = np.append(pathMap, origin, axis=1)
    for i in range(len(gtPose)-1):
        poseA = kitti._get_3x4_tmat(gtPose[i])
        poseB = kitti._get_3x4_tmat(gtPose[i+1])
        pose = _get_tMat_A_2_B(poseA, poseB)
        origin = kitti.transform_pcl(origin, pose)
        pathMap = np.append(pathMap, origin, axis=1)
    return pathMap
def _get_gt_map(gtPose):
    """
    get the ground truth path map
    pose are w.r.t. the origin
    """
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    pathMap = np.append(pathMap, origin, axis=1)
    for i in range(len(gtPose)):
        pose = kitti._get_3x4_tmat(gtPose[i])
        pointT = kitti.transform_pcl(origin, pose)
        pathMap = np.append(pathMap, pointT, axis=1)
    return pathMap

def _get_p_map(pPose):
    """
    get the predicted truth path map
    poses are w.r.t. previous frame
    """
    origin = np.array([[0], [0], [0]], dtype=np.float32)
    pathMap = np.ndarray(shape=[3,0], dtype=np.float32)
    pathMap = np.append(pathMap, origin, axis=1)
    for i in range(len(pPose)):
        pose = kitti._get_3x4_tmat(np.array(pPose[i]['tmat']))
        origin = kitti.transform_pcl(origin, pose)
        pathMap = np.append(pathMap, origin, axis=1)
    return pathMap

def _get_control_params():
    """
    Get control parameters for the specific task
    """
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
    pFilenames = _get_file_names(modelParams['tMatDir'], "")
    predPoses = _get_all_predictions(pFilenames)
    # For each sequence
    for i in range(len(modelParams['seqIDs'])):
        print("Processing sequences: {0} / {1}".format(i, len(modelParams['seqIDs'])))
        # Read groundtruth posefile for a seqID
        # create map
        gtPosePath = kitti.get_pose_path(modelParams['gTruthDir'], modelParams['seqIDs'][i])
        gtPose = kitti.get_pose_data(gtPosePath)
        gtMap = _get_gt_map(gtPose) # w.r.t. Original
        gtMap = _get_gt_map_seq(gtPose) # w.r.t. sequential
        # Get predictions for a seqID
        # create map
        pPose = _get_prediction(predPoses, modelParams['seqIDs'][i])
        pMap = _get_p_map(pPose) # w.r.t. sequential
        vis_path(gtMap, pMap)

################################
def vis_path(gtxyz, pxyz):
    import matplotlib.pyplot as plt
    plt.plot(gtxyz[0], gtxyz[1], 'r', pxyz[0], pxyz[1])
    plt.show()


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
    #if input("IS PRESENTED INFORMATION VALID? ") != "yes":
    #    print("Please consider updating the provided information!")
    #    return
    evaluate()

main()