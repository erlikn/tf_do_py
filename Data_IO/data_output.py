# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
import os
import json
import collections
import math
import random
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

import Data_IO.tfrecord_io as tfrecord_io
import Data_IO.kitti_shared as kitti

def _apply_prediction(pclA, tMatT, tMatP, **kwargs):
    '''
    Transform pclA, Calculate new tMatT based on tMatP, Create new depth image
    Return:
        - New PCLA
        - New tMatT
        - New depthImage
    '''
    # remove trailing zeros
    pclA = kitti.remove_trailing_zeros(pclA)
    # get transformed pclA based on tMatP
    pclATransformed = kitti.transform_pcl(pclA, tMatP)
    # get new depth image of transformed pclA
    depthImageA, _ = kitti.get_depth_image_pano_pclView(pclATransformed)
    pclATransformed = kitti._zero_pad(pclATransformed, kwargs.get('pclCols')-pclATransformed.shape[1])
    # get residual tMat
    tMatResA2B = kitti.get_residual_tMat_A2B(tMatT, tMatP)
    return pclATransformed, tMatResA2B, depthImageA

def output(batchImages, batchPclA, batchPclB, batchtMatT, batchtMatP, batchTFrecFileIDs, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    for i in range(kwargs.get('activeBatchSize')):
        # split for depth dimension
        depthA, depthB = np.asarray(np.split(batchImages[i], 2, axis=2))
        depthB = depthB.reshape(kwargs.get('imageDepthRows'), kwargs.get('imageDepthCols'))
        pclATransformed, tMatRes, depthATransformed = _apply_prediction(batchPclA[i], batchtMatT[i], batchtMatP[i], **kwargs)
        # Write each Tensorflow record
        filename = str(batchTFrecFileIDs[i][0]) + "_" + str(batchTFrecFileIDs[i][1]) + "_" + str(batchTFrecFileIDs[i][2])
        tfrecord_io.tfrecord_writer(batchTFrecFileIDs[i],
                                    pclATransformed, batchPclB[i],
                                    depthATransformed, depthB,
                                    tMatRes,
                                    kwargs.get('warpedOutputFolder')+'/', filename)
        if kwargs.get('phase') == 'train':
            folderTmat = kwargs.get('trainLogDir')+'/'+'tmat'
        else:
            folderTmat = kwargs.get('testLogDir')+'/'+'tmat'
        write_predictions(batchTFrecFileIDs[i], batchtMatP[i], folder_tmat)
    return

def write_json_file(filename, datafile):
    filename = 'Model_Settings/../'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent = 0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def write_predictions(tfrecID, tmatP, folderOut):
    """
    Write prediction outputs to generate path map
    """
    _set_folders(folderOut)
    dataJson = {'tmat' : tmatP}
    write_json_file(folderOut + '/' + str(tfrecID[0]) + '_' + str(tfrecID[1]) + str(tfrecID[2]), dataJson)
    return
