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

def perturb_writer( ID, idx,
                    imgOrig, imgPatchOrig, imgPatchPert, HAB, pOrig,
                    tfRecFolder):
    filename = str(ID) + "_" + str(idx)
    fileID = [ID, idx]
    tfrecord_io.tfrecord_writer(imgOrig, imgPatchOrig, imgPatchPert, pOrig, HAB, tfRecFolder, filename, fileID)
    return

################################
def transform_pcl(xyzi_col, tMat):
    i_col = xyzi_col[3]
    xyz1 = xyzi_col.copy()
    xyz1[3] *= 0
    xyz1[3] += 1
    xyz1 = np.matmul(tMat, xyz1)
    xyz1[3] = i_col
    return xyz1
################################
def _get_correct_tmat(poseRow):
    return (np.array(np.append(poseRow, [0, 0, 0, 1]), dtype=np.float32)).reshape([4,4])
################################
def _get_pcl(filePath):
    f = open(filePath, 'rb')
    i = 0
    pclpoints = list()
    # Camera: x = right, y = down, z = forward
    # Velodyne: x = forward, y = left, z = up
    # GPS/IMU: x = forward, y = left, z = up
    # Velodyne -> Camera (transformation matrix is in camera order)
    #print('Reading X = -y, Y = -z, Z = x, i = 3')
    while f.readable():
        xyzi = f.read(4*4)
        if len(xyzi) == 16:
            row = struct.unpack('f'*4, f.read(4*4))
            pclpoints.append([-1*row[1], -1*row[2], row[0], row[3]])
            i += 1
        else:
            break
        #if i == 15000:
        #    break
    f.close()
    # convert to numpy
    xyzi = np.array(pclpoints, dtype=np.float32)
    return xyzi.transpose()
################################
def process_dataset(startTime, durationSum, pclFilenames, pclFilenames, tfRecFolder, poseFile, i):
    '''
    point cloud is moved to i'th frame
    tmat_0r, tmat_1r => tmat10 = tmat_1r*inv(tmat_0r)
    '''
    # get i
    xyzi_0 = _get_pcl(pclFilenames[i])
    if i==0:
        pose_0 = np.identity(4)
    else:
        pose_0 = _get_correct_tmat(poseFile[i])
    imgDepth_0 = _get_depthMap(xyzi_0)
    # get i+1
    xyzi_1 = _get_pcl(pclFilenames[i+1])
    pose_1 = _get_correct_tmat(poseFile[i+1])
    imgDepth_1 = _get_depthMap(xyzi_1)
    pose_10 = np.matmul(pose_1, np.linalg.inv(pose_0))
    fileID = [i-1, i]
    perturb_writer(fileID,
                   xyzi_0, xyzi_1,
                   imgDepth_0, imgDepth_1,
                   pose_10,
                   tfRecFolder)


################################
def _get_pose_data(posePath):
    return np.loadtxt(open(posePath, "r"), delimiter=" ", skiprows=1)
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
        Parallel(n_jobs=num_cores)(delayed(process_dataset)(startTime, durationSum, pclFilenames, pclFilenames, tfRecFolder, poseFile, i) for i in range(len(pclFilenames-1)))
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
