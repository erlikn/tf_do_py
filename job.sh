#!/bin/sh
python3 write_tfrecords_ntuple.py 180521_170706 3
python3 train_main_ntuple.py 180523_170706 1
python3 write_tfrecords_ntuple.py 180523_170706 1
