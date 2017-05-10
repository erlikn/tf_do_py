# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from datetime import datetime
import time
import math
import random
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

import struct
from scipy import spatial

from joblib import Parallel, delayed
import multiprocessing

import tfrecord_io

def image_process_subMean_divStd(img):
    out = img - np.mean(img)
    out = out / img.std()
    return out

def image_process_subMean_divStd_n1p1(img):
    out = img - np.mean(img)
    out = out / img.std()
    out = (2*((out-out.min())/(out.max()-out.min())))-1
    return out

def perturb_writer( ID,
                    pclA, pclB,
                    imgDepthA, imgDepthB,
                    tMatTarget,
                    tfRecFolder):
    '''
    ID: python list with size 2
    pclA, pclB: numpy matrix of size nx4, mx4
    imgDepthA, imgDepthB: numpy matrix of size 128x512
    tmatTarget: numpy matrix of size 4x4
    tfRecFolder: folder name
    '''
    filename = str(ID) + "_" + str(idx)
    fileID = [ID, idx]
    tfrecord_io.tfrecord_writer(imgOrig, imgPatchOrig, imgPatchPert, pOrig, HAB, tfRecFolder, filename, fileID)
    return

################################
def transform_pcl_2_origin(xyzi_col, tMat2o):
    '''
    pointcloud i, and tMat2o i to origin
    '''
    intensity_col = xyzi_col[3]
    xyz1 = xyzi_col.copy()
    xyz1[3] *= 0
    xyz1[3] += 1
    xyz1 = np.matmul(tMat2o, xyz1)
    xyz1[3] = intensity_col
    return xyz1
################################
def _get_tMat_A_2_B(tMatA2o, tMatB2o):
    '''
    tMatA2o A -> O (source pcl is in A), tMatB2o B -> O (target pcl will be in B)
    return tMat A -> B
    '''
    # tMatA2o: A -> Orig
    # tMatB2o: B -> Orig ==> inv(tMatB2o): Orig -> B
    # inv(tMatB2o) * tMatA2o : A -> B
    return np.matmul(np.matmul(np.linalg.inv(tMatB2o), tMatA2o)
################################
def _get_correct_tmat(poseRow):
    return (np.array(np.append(poseRow, [0, 0, 0, 1]), dtype=np.float32)).reshape([4,4])
################################
def _get_pcl(filePath):
    '''
    Get a bin file address and read it into a numpy matrix
    Converting LiDAR coordinate system to Camera coordinate system for pose transform
    '''
    f = open(filePath, 'rb')
    i = 0
    j = 0
    pclpoints = list()
    # Camera: x = right, y = down, z = forward
    # Velodyne: x = forward, y = left, z = up
    # GPS/IMU: x = forward, y = left, z = up
    # Velodyne -> Camera (transformation matrix is in camera order)
    #print('Reading X = -y,         Y = -z,      Z = x,     i = 3')
    #               0 = left/right, 1 = up/down, 2 = in/out
    while f.readable():
        xyzi = f.read(4*4)
        if len(xyzi) == 16:
            row = struct.unpack('f'*4, xyzi)
            if j%1 == 0:
                pclpoints.append([-1*row[1], -1*row[2], row[0], row[3]])
                i += 1
        else:
            print('num pclpoints =', i)
            break
        j += 1
        #if i == 15000:
        #    break
    f.close()
    # convert to numpy
    xyzi = np.array(pclpoints, dtype=np.float32)
    return xyzi.transpose()

################################
def process_dataset(startTime, durationSum, pclFilenames, poseFile, tfRecFolder, i):
    '''
    pclFilenames: list of pcl file addresses
    poseFile: includes a list of pose files to read
    point cloud is moved to i+1'th frame:
        tMatAo (i): A->0
        tMatBo (i+1): B->0
        tMatAB (target): A->B  (i -> i+1) 
    '''
    # get i
    xyzi_A = _get_pcl(pclFilenames[i])
    pose_Ao = _get_correct_tmat(poseFile[i])
    imgDepth_A = _get_depthMap(xyzi_A)
    # get i+1
    xyzi_B = _get_pcl(pclFilenames[i+1])
    pose_Bo = _get_correct_tmat(poseFile[i+1])
    imgDepth_B = _get_depthMap(xyzi_B)
    # get target pose
    pose_AB = _get_tMat_A_2_B(pose_Ao, pose_Bo)
    #
    fileID = [i-1, i]
    perturb_writer(fileID,
                   xyzi_A, xyzi_B,
                   imgDepth_A, imgDepth_B,
                   pose_AB,
                   tfRecFolder)

################################
def _get_pose_data(posePath):
    return np.loadtxt(open(posePath, "r"), delimiter=" ")
################################
def _get_pcl_folder(pclFolder, seqID):
    return pclFolder + seqID + '/' + 'velodyne/'
################################
def _get_pose_path(poseFolder, seqID):
    return poseFolder + seqID + ".txt"
################################
def _get_file_names(readFolder):
    return [f for f in listdir(readFolder) if isfile(join(readFolder, f))]
################################
def prepare_dataset(datasetType, readPath, seqIDs, tfRecFolder):
    durationSum = 0
    for i in range(len(seqIDs)):
        print('Procseeing ', seqIDs[i])
        poseFile = _get_pose_data(_get_pose_path(readPath, seqIDs[i]))
        pclFilenames = _get_file_names(_get_pcl_folder(readPath, seqIDs[i]))
        pclFilenames.sort()
        startTime = time.time()
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(process_dataset)(startTime, durationSum, pclFilenames, poseFile, tfRecFolder, i) for i in range(len(pclFilenames-1)))
    print('Done')

############# PATHS
dataPath = '../Data/kitti/pointcloud/'
posePath = '../Data/kitti/visual/poses/'
seqIDtrain = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
seqIDtest = ['09', '10']

traintfRecordFLD = "../../Data/train_tfrecords/"
testtfRecordFLD = "../../Data/test_tfrecords/"

prepare_dataset("test", dataPath, posePath, seqIDtest, testtfRecordFLD)
prepare_dataset("train", dataPath, posePath, seqIDtrain, traintfRecordFLD)
