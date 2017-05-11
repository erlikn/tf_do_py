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

def _get_pcl(pcl):
    """
    Decode and put point cloud in the right form. nx4
    """
    pcl = tf.reshape(features['pclA'], [-1, 4])
    #pcl.set_shape([-1, 4])
    return pcl

def parse_example_proto(exampleSerialized, **kwargs):
    """
        ID: python list with size 2
        pclA: numpy matrix of size mx4
        pclB: numpy matrix of size nx4
        image: numpy matrix of 128x512x2
            imgDepthA: numpy matrix of size 128x512
            imgDepthB: numpy matrix of size 128x512
        tMatTarget: numpy matrix of size 4x4

        'ID': _int64_feature(IDList),
        'pclA': _float_nparray(pclAList),
        'pclB': _float_nparray(pclBList),
        'image': _bytes_feature(flatImageList)
        'tMatTarget': _float_nparray(tMatist), # 2D np array
    """

    featureMap = {
        'fileID': tf.FixedLenFeature([2], dtype=tf.int64),
        'images': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'pclA': tf.FixedLenFeature([], dtype=tf.float32, default_value=''),
        'pclB': tf.FixedLenFeature([], dtype=tf.float32, default_value=''),
        'tMatTarget': tf.FixedLenFeature([8], dtype=tf.float32)
        }
    features = tf.parse_single_example(exampleSerialized, featureMap)

    fileID = features['fileID']
    images = _decode_byte_image(features['images'],
                                kwargs.get('imageDepthHeight'),
                                kwargs.get('imageDepthWidth'),
                                kwargs.get('imageDepthChannels'))
    pclA = _get_pcl(features['pclA'])
    pclB = _get_pcl(features['pclB'])
    tMat = features['tMatTarget']
    return images, pclA, pclB, tMat, fileID

def tfrecord_writer(fileID,
                    pclA, pclB,
                    imgDepthA, imgDepthB,
                    tMatTarget,
                    tfRecFolder, tfFileName):
    """
    Converts a dataset to tfrecords
    imgDepthA, imgDepthB => int8 a.k.a. char
    tMatTarget => will be converted to float32
    pclA, pclB => will be converted to float16
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
    pclA = pclA.reshape(pclA.shape[0]*pclA.shape[1]) # nx4
    pclAlist = pclA.tolist()
    pclB = pclB.reshape(pclB.shape[0]*pclB.shape[1]) # mx4
    pclBlist = pclB.tolist()
    # Target Transformation
    tMatTarget = tMatTarget.reshape(tMatTarget.shape[0]*tMatTarget.shape[1]) # 4x4
    tMatTargetList = tMatTarget.tolist()

    writer = tf.python_io.TFRecordWriter(tfRecordPath)
    example = tf.train.Example(features=tf.train.Features(feature={
        'fileID': _int64_array(fileID),
        'images': _bytes_feature(flatImageList),
        'pclA': _float_nparray(pclAlist), # 2D np array
        'pclB': _float_nparray(pclBlist), # 2D np array
        'tMatTarget': _float_nparray(tMatTargetList) # 2D np array
        }))
    writer.write(example.SerializeToString())
    writer.close()


