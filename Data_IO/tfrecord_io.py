# Python 3.4
import sys
sys.path.append("/usr/local/lib/python3.4/site-packages/")
import cv2 as cv2
from os import listdir
from os.path import isfile, join
from os import walk
from shutil import copy
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# to get HAB and pOrig
def _float_nparray(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _decode_byte_string(filename):
    """Decode and preprocess one filename.
    Args:
      filename: Binary string Tensor
    Returns:
      String Tensor containing the image in float32
    """
    tfname = tf.decode_raw(filename, tf.uint8)
    return tfname

def _decode_byte_image(image, height, width, depth):
    """Decode and preprocess one image for evaluation or training.
    Args:
      imageBuffer: Binary string Tensor
      Height, Widath, Channels <----- GLOBAL VARIABLES ARE USED DUE TO SET_SHAPE REQUIREMENTS
    Returns:
      3-D float Tensor containing the image in float32
    """
    image = tf.decode_raw(image, tf.float32)
    if depth > 1:
        image = tf.reshape(image, [height, width, depth])
        image.set_shape([height, width, depth])
    else:
        image = tf.reshape(image, [height, width])
        image.set_shape([height, width])
    return image

def _get_pcl(pcl, rows, cols):
    """
    Decode and put point cloud in the right form. nx4
    """
    #pcl = tf.decode_raw(pcl, tf.float32)
    pcl = tf.reshape(pcl, [rows, cols])
    pcl.set_shape([rows, cols])
    return pcl

def parse_example_proto(exampleSerialized, **kwargs):
    """
        ID: python list with size 2
        pclA: numpy matrix of size 3xPCL_COLS
        pclB: numpy matrix of size 3xPCL_COLS
        image: numpy matrix of 128x512x2
            imgDepthA: numpy matrix of size 128x512
            imgDepthB: numpy matrix of size 128x512
        targetABGXYZ: numpy matrix of size 3x4 = 12

        'ID': _int64_feature(IDList),
        'pclA': _float_nparray(pclAList),
        'pclB': _float_nparray(pclBList),
        'image': _bytes_feature(flatImageList)
        'targetABGXYZ': _float_nparray(targetList), # 2D np array
    """
    """
    KWARGS:
        imageDepthRows = 128
        imageDepthCols = 512
        imageDepthChannels = 2

        pclRows = 3
        pclCols = 62074

        targetABGXYZ = 6
    """
    featureMap = {
        'fileID': tf.FixedLenFeature([3], dtype=tf.int64),
        'images': tf.FixedLenFeature([], dtype=tf.string),
        'pclA': tf.FixedLenFeature([kwargs.get('pclRows')*kwargs.get('pclCols')], dtype=tf.float32),
        'pclB': tf.FixedLenFeature([kwargs.get('pclRows')*kwargs.get('pclCols')], dtype=tf.float32),
        'targetABGXYZ': tf.FixedLenFeature([6], dtype=tf.float32)
        }
    features = tf.parse_single_example(exampleSerialized, featureMap)
    fileID = features['fileID']
    images = _decode_byte_image(features['images'],
                                kwargs.get('imageDepthRows'),
                                kwargs.get('imageDepthCols'),
                                kwargs.get('imageDepthChannels'))
    pclA = _get_pcl(features['pclA'], kwargs.get('pclRows'), kwargs.get('pclCols'))
    pclB = _get_pcl(features['pclB'], kwargs.get('pclRows'), kwargs.get('pclCols'))
    target = features['targetABGXYZ']

    # PCLs will hold padded [0, 0, 0, 0] points at the end that will be ignored during usage
    # However, they will be kept to unify matrix col size for valid tensor operations 
    return images, pclA, pclB, target, fileID

def tfrecord_writer(fileID,
                    pclA, pclB,
                    imgDepthA, imgDepthB,
                    targetABGXYZ,
                    tfRecFolder, tfFileName):
    """
    Converts a dataset to tfrecords
    fileID = seqID, i, i+1
    imgDepthA, imgDepthB => int8 a.k.a. char 128x512
    targetABGXYZ => will be converted to float32 with size 6
    pclA, pclB => will be converted to float16 with size 3xPCLCOLS
    """
    tfRecordPath = tfRecFolder + tfFileName + ".tfrecords"
    # Depth Images
    rows = imgDepthA.shape[0]
    cols = imgDepthA.shape[1]
    depth = 2
    stackedImage = np.stack((imgDepthA, imgDepthB), axis=2) #3D array (hieght, width, channels)
    flatImage = stackedImage.reshape(rows*cols*depth)
    flatImage = np.asarray(flatImage, np.float32)
    flatImageList = flatImage.tostring()
    # Point Clouds
    pclA = pclA.reshape(pclA.shape[0]*pclA.shape[1]) # 3 x PCL_COLS
    pclAlist = pclA.tolist()
    pclB = pclB.reshape(pclB.shape[0]*pclB.shape[1]) # 3 x PCL_COLS
    pclBlist = pclB.tolist()
    # Target Transformation
    targetABGXYZList = targetABGXYZ.tolist()

    writer = tf.python_io.TFRecordWriter(tfRecordPath)
    example = tf.train.Example(features=tf.train.Features(feature={
        'fileID': _int64_array(fileID),
        'images': _bytes_feature(flatImageList),
        'pclA': _float_nparray(pclAlist), # 2D np array
        'pclB': _float_nparray(pclBlist), # 2D np array
        'targetABGXYZ': _float_nparray(targetABGXYZList) # 2D np array
        }))
    writer.write(example.SerializeToString())
    writer.close()


