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

def _apply_prediction(pclA, tMatT, tMatP):
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
    depthImageA = kitti.get_depth_image_pano_pclView(pclATransformed)
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
        pclATransformed, tMatRes, depthATransformed = _apply_prediction(batchPclA[i], batchtMatT[i], batchtMatP[i])
        # Write each Tensorflow record
        filename = str(batchTFrecFileIDs[i][0]) + "_" + str(batchTFrecFileIDs[i][1]) + "_" + str(batchTFrecFileIDs[i][2])
        tfrecord_io.tfrecord_writer(batchTFrecFileIDs[i],
                                    pclATransformed, batchPclB[i],
                                    depthATransformed, depthB,
                                    tMatRes,
                                    kwargs.get('warpedOutputFolder')+'/', filename)
    return

def write_json_file(filename, datafile):
    filename = 'Model_Settings/../'+filename
    datafile = collections.OrderedDict(sorted(datafile.items()))
    with open(filename, 'w') as outFile:
        json.dump(datafile, outFile, indent = 0)

def _set_folders(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def output_with_test_image_files(batchImageOrig, batchImage, batchPOrig, batchTHAB, batchPHAB, batchTFrecFileIDs, **kwargs):
    """
    TODO: SIMILAR TO DATA INPUT -> WE NEED A QUEUE RUNNER TO WRITE THIS OFF TO BE FASTER 

    Everything evaluated
    Warp second image based on predicted HAB and write to the new address
    Args:
    Returns:
    Raises:
      ValueError: If no dataDir
    """
    imagesOutputFolder = kwargs.get('testLogDir')+'/images/'
    _set_folders(imagesOutputFolder)
    ## for each value call the record writer function
    corrupt_patchOrig = 0
    corrupt_patchPert = 0
    corrupt_imageOrig = 0
    dataJson = {'pOrig':[], 'tHAB':[], 'pHAB':[]}
    for i in range(kwargs.get('activeBatchSize')):
        dataJson['pOrig'] = batchPOrig[i].tolist()
        dataJson['tHAB'] = batchTHAB[i].tolist()
        dataJson['pHAB'] = batchPHAB[i].tolist()
        write_json_file(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1]), dataJson)
        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+ str(batchTFrecFileIDs[i][1])+'_fullOrig.jpg', batchImageOrig[i]*255)

        orig, pert = np.asarray(np.split(batchImage[i], 2, axis=2))
        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1])+'_ob.jpg', orig*255)
        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1])+'_pb.jpg', pert*255)

        # Get the difference of tHAB and pHAB, and make new perturbed image based on that
        cHAB = batchTHAB[i]-batchPHAB[i]
        # put them in correct form
        HAB = np.asarray([[cHAB[0], cHAB[1], cHAB[2], cHAB[3]],
                          [cHAB[4], cHAB[5], cHAB[6], cHAB[7]]], np.float32)
        pOrig = np.asarray([[batchPOrig[i][0], batchPOrig[i][1], batchPOrig[i][2], batchPOrig[i][3]],
                            [batchPOrig[i][4], batchPOrig[i][5], batchPOrig[i][6], batchPOrig[i][7]]])
        if kwargs.get('warpOriginalImage'):
            patchOrig, patchPert = _warp_w_orig_newTarget(batchImageOrig[i], batchImage[i], pOrig, HAB, **kwargs)
            # NOT DEVELOPED YET
            #imageOrig, imagePert = _warp_w_orig_newOrig(batchImageOrig[i], batchImage[i], pOrig, batchPHAB[i], **kwargs)
        else:
            patchOrig, patchPert = _warp_wOut_orig_newTarget(batchImage[i], batchPHAB[i])

        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1])+'_op.jpg', patchOrig*255)
        cv2.imwrite(imagesOutputFolder+str(batchTFrecFileIDs[i][0])+'_'+str(batchTFrecFileIDs[i][1])+'_pp.jpg', patchPert*255)
        # Write each Tensorflow record
        fileIDs = str(batchTFrecFileIDs[i][0]) + '_' + str(batchTFrecFileIDs[i][1])
        tfrecord_io.tfrecord_writer(batchImageOrig[i], patchOrig, patchPert, pOrig, HAB,
                                    kwargs.get('warpedOutputFolder')+'/',
                                    fileIDs, batchTFrecFileIDs[i])
        if batchImageOrig[i].shape[0] != 240:
            corrupt_imageOrig+=1
        if patchOrig.shape[0]!=128:
            corrupt_patchOrig+=1
        if patchPert.shape[0]!=128:
            corrupt_patchPert+=1

    return corrupt_imageOrig, corrupt_patchOrig, corrupt_patchPert