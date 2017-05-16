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
#function img = make_image(depthview)
#    %depthview(:,1) = depthview(:,1)*(-1);
#    %depthview(:,2) = depthview(:,2)*(-1);
#    % 0 - max (not necesseary)
#    depthview(:,1) = depthview(:,1) + min(depthview(:,1));
#    depthview(:,2) = depthview(:,2) + min(depthview(:,2));
#    % there roughly should be 64 height bins group them in 64 clusters
#    [yHist, yCent] = hist(depthview(:,2), 64);
#    [xHist, xCent] = hist(depthview(:,1), 512);
#    % make image of same size
#    img = zeros(64, 512);
#    % normalize range values
#    % depthview(:,3)=depthview(:,3)*(-1) % if back view 
#    depthview=[depthview; 0,0,0];
#    depthview=[depthview; 0,0,100];
#    depthview(:,3) = mat2gray(depthview(:,3));
#    depthview(:,3) = 1-depthview(:,3);
#    depthview = depthview(1:end-2,:);
#    [values, order] = sort(depthview(:,3),'descend');
#    depthview = depthview(order,:); ;;;
#    % assign range to pixels
#    xidxv = [];
#    yidxv = [];
#    for i=1:size(depthview,1)
#        [diff, yidx] = min(abs(yCent-depthview(i,2)));
#        [diff, xidx] = min(abs(xCent-depthview(i,1)));
#        yidx=yidx*2;
#        xidxv = [xidxv xidx];



#        yidxv = [yidxv yidx];
#        img(yidx, xidx) = depthview(i,3);
#        img(yidx+1, xidx) = depthview(i,3);
#    end
#end
def _make_image(depthview):
    '''
    Get depthview and generate a depthImage
    '''
    depthview = np.append(depthview, [[0,0],[0,0],[0,100]], axis=1)
    # 0 - max (not necesseary)
    depthview[0] = depthview[0] - np.min(depthview[0])
    depthview[1] = depthview[1] - np.min(depthview[1])
    # there roughly should be 64 height bins group them in 64 clusters
    xHist, xBinEdges = np.histogram(depthview[0], 512)
    yHist, yBinEdges = np.histogram(depthview[1], 64)
    xCent = np.ndarray(shape=xBinEdges.shape[0]-1)
    for i in range(0, xCent.shape[0]):
        xCent[i] = (xBinEdges[i]+xBinEdges[i+1])/2
    yCent = np.ndarray(shape=yBinEdges.shape[0]-1)
    for i in range(0, yCent.shape[0]):
        yCent[i] = (yBinEdges[i]+yBinEdges[i+1])/2
    # make image of size 128x512 : 64 -> 128 (double sampling the height)
    depthImage = np.zeros(shape=[128, 512])
    # normalize range values
    depthview[2] = (depthview[2]-np.min(depthview[2]))/(np.max(depthview[2])-np.min(depthview[2]))
    depthview[2] = 1-depthview[2]
    depthview = np.delete(depthview, depthview.shape[1]-1,1)
    depthview = np.delete(depthview, depthview.shape[1]-1,1)
    # sorts ascending
    idxs = np.argsort(depthview[2], kind = 'mergesort')
    # assign range to pixels
    for i in range(depthview.shape[1]-1,-1,-1): # traverse descending
        yidx = np.argmin(np.abs(yCent-depthview[1,i]))
        xidx = np.argmin(np.abs(xCent-depthview[0,i]))
        # hieght is 2x64
        yidx=yidx*2
        depthImage[yidx, xidx] = depthview[2,i]
        depthImage[yidx+1, xidx] = depthview[2,i]
    return depthImage

################################
def get_depth_image_pano_pclView(xyzi, height=1.6):
    '''
    Gets a point cloud
    Keeps points higher than 'height' value and located on the positive Z=0 plane
    Returns corresponding depthMap and pclView
    '''
    #print('0', max(xyzi[0]), min(xyzi[0])) # left/right (-)
    #print('1', max(xyzi[1]), min(xyzi[1])) # up/down (-)
    #print('2', max(xyzi[2]), min(xyzi[2])) # in/out
    xyzi = xyzi.transpose()
    first = True
    for i in range(xyzi.shape[0]):
        # xyzi[i][2] > 0 means all the points who have depth larger than 0 (positive depth plane)
        if xyzi[i][2] > 0 and xyzi[i][1] < height: # frontal view, above ground
            rXZ = np.sqrt(
                          np.multiply(xyzi[i][0], xyzi[i][0])+
                          np.multiply(xyzi[i][2], xyzi[i][2]))
            if (xyzi[i][1]/rXZ < 0.5) and (xyzi[i][1]/rXZ > -0.1): # view planes (above ground y=-az) and (below highest y=az)
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
    #depthImage = _make_image(np.append(np.append(xT, yT, axis=0), zT, axis=0))
    depthImage = 0
    return depthImage, pclview
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
    print(pclFolder + pclFilenames[i])
    cv2.imshow('img', imgDepth_A)
    cv2.waitKey(1)
    return
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
def prepare_dataset(datasetType, pclFolder, poseFolder, seqIDs, tfRecFolder):
    durationSum = 0
    for i in range(len(seqIDs)):
        print('Procseeing ', seqIDs[i])
        posePath = _get_pose_path(poseFolder, seqIDs[i])
        poseFile = _get_pose_data(posePath)
        print(posePath)

        pclFolderPath = _get_pcl_folder(pclFolder, seqIDs[i])
        pclFilenames = _get_file_names(pclFolderPath)
        startTime = time.time()
        num_cores = multiprocessing.cpu_count()
        maxs = np.array([0.0,0.0,0.0,0.0,0.0], np.float32)
        mins = np.array([0.0,0.0,0.0,0.0,0.0], np.float32)
        count = 0
        for i in range(0,100):#len(pclFilenames)-1):
            process_dataset(startTime, durationSum, pclFolderPath, pclFilenames, poseFile, tfRecFolder, i)
        #Parallel(n_jobs=num_cores)(delayed(process_dataset)(startTime, durationSum, pclFilenames, poseFile, tfRecFolder, i) for i in range(len(pclFilenames)-1))
    print('Done')

############# PATHS
pclPath = '../Data/kitti/pointcloud/'
posePath = '../Data/kitti/poses/'
seqIDtrain = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
seqIDtest = ['09', '10']

traintfRecordFLD = "../Data/train_tfrecords/"
testtfRecordFLD = "../Data/test_tfrecords/"

prepare_dataset("train", pclPath, posePath, seqIDtrain, traintfRecordFLD)
#prepare_dataset("test", pclPath, posePath, seqIDtest, testtfRecordFLD)
