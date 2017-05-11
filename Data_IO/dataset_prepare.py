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
    filename = str(ID[0]) + "_" + str(ID[1])
    tfrecord_io.tfrecord_writer(fileID,
                                pclA, pclB,
                                imgDepthA, imgDepthB,
                                tMatTarget,
                                tfRecFolder, filename)
    return

################################
def get_depth_image_pano_pclView(xyzi, height=1.6):
    '''
    Gets a point cloud
    Keeps points higher than 'height' value and located on the positive Z=0 plane
    Returns corresponding depthMap and pclView
    '''
    print('0', max(xyzi[0]), min(xyzi[0])) # left/right (-)
    print('1', max(xyzi[1]), min(xyzi[1])) # up/down (-)
    print('2', max(xyzi[2]), min(xyzi[2])) # in/out
    xyzi = xyzi.transpose()
    first = True
    for i in range(xyzi.shape[0]):
        # xyzi[i][2] > 0 means all the points who have depth larger than 0 (positive depth plane)
        if xyzi[i][2] > 0 and xyzi[i][1] < height:
            if first:
                pclview = xyzi[i].reshape(1, 4)
                first = False
            else:
                pclview = np.append(pclview, xyzi[i].reshape(1, 4), axis=0)
    xyzi = xyzi.transpose()
    pclview = pclview.transpose()
    rXYZ = np.sqrt(
                   np.multiply(pclview[0], pclview[0])+
                   np.multiply(pclview[1], pclview[1])+
                   np.multiply(pclview[2], pclview[2]))
    # 0 left-right, 1 is up-down, 2 is forward-back
    xT = (pclview[0]/rXYZ).reshape([1, pclview.shape[1]])
    yT = (pclview[1]/rXYZ).reshape([1, pclview.shape[1]])
    zT = rXYZ.reshape([1, pclview.shape[1]])
    return np.append(np.append(xT, yT, axis=0), zT, axis=0), pclview
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
    return np.matmul(np.matmul(np.linalg.inv(tMatB2o), tMatA2o))

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
def process_dataset(startTime, durationSum, pclFolder, pclFilenames, poseFile, tfRecFolder, i):
    '''
    pclFilenames: list of pcl file addresses
    poseFile: includes a list of pose files to read
    point cloud is moved to i+1'th frame:
        tMatAo (i): A->0
        tMatBo (i+1): B->0
        tMatAB (target): A->B  (i -> i+1) 
    '''
    # get i
    xyzi_A = _get_pcl(pclFolder + pclFilenames[i])
    pose_Ao = _get_correct_tmat(poseFile[i])
    imgDepth_A, xyzi_A = get_depth_image_pano_pclView(xyzi_A)
    # get i+1
    xyzi_B = _get_pcl(pclFilenames[i+1])
    pose_Bo = _get_correct_tmat(poseFile[i+1])
    imgDepth_B, xyzi_B = get_depth_image_pano_pclView(xyzi_B)
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
    filenames = [f for f in listdir(readFolder) if isfile(join(readFolder, f))]
    filenames.sort()
    return filenames
################################
def prepare_dataset(datasetType, pclPath, posePath, seqIDs, tfRecFolder):
    durationSum = 0
    for i in range(len(seqIDs)):
        print('Procseeing ', seqIDs[i])
        poseFolder = _get_pose_path(posePath, seqIDs[i])
        poseFile = _get_pose_data(poseFolder)
        pclFolder = _get_pcl_folder(pclPath, seqIDs[i])
        pclFilenames = _get_file_names(pclFolder)
        startTime = time.time()
        num_cores = multiprocessing.cpu_count()
        for i in range(len(pclFilenames)-1):
            process_dataset(startTime, durationSum, pclFolder, pclFilenames, poseFile, tfRecFolder, i)
        #Parallel(n_jobs=num_cores)(delayed(process_dataset)(startTime, durationSum, pclFilenames, poseFile, tfRecFolder, i) for i in range(len(pclFilenames)-1))
    print('Done')

############# PATHS
pclPath = '../Data/kitti/pointcloud/'
posePath = '../Data/kitti/poses/'
seqIDtrain = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
seqIDtest = ['09', '10']

traintfRecordFLD = "../Data/train_tfrecords/"
testtfRecordFLD = "../Data/test_tfrecords/"

prepare_dataset("test", pclPath, posePath, seqIDtest, testtfRecordFLD)
prepare_dataset("train", pclPath, posePath, seqIDtrain, traintfRecordFLD)
