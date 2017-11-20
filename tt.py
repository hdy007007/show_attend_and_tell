#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:57:35 2017

@author: root
"""
from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json
import jieba



batch_size = 50
vggnet = Vgg19('./ai.challenger/data/imagenet-vgg-verydeep-19.mat')
vggnet.build()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#with tf.Session(config = config) as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    for split in ['train']:
#        anno_path = './data/%s/%s.annotations.pkl' % (split, split)
#        
#        annotations = load_pickle(anno_path)
#        image_path = list(annotations['file_name'].unique())
#        n_examples = len(image_path)
#        
#        for i in [4]:
#            feats_part = np.ndarray([n_examples - i*50000, 196, 512], dtype=np.float32)
#            for start, end in zip(range(i*50000, n_examples, batch_size),
#                                  range(i*50000 + batch_size, n_examples + batch_size, batch_size)):
#                image_batch_file = image_path[start:end]
#                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
#                    np.float32)
#                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
#                feats_part[start - i*50000: end - i*50000, :] = feats
#                print ("Processed %d %s%dfeatures.." % (end, split,i + 1))
#
#            save_path = './data/%s/%s.features%d.hkl' % (split, split,i + 1)
#            # use hickle to save huge feature vectors
#            hickle.dump(feats_part, save_path)
#            print ("Saved %s.." % (save_path))

#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    for split in [ 'val', 'test']:
#        anno_path = './data/%s/%s.annotations.pkl' % (split, split)
#        save_path = './data/%s/%s.features.hkl' % (split, split)
#        annotations = load_pickle(anno_path)
#        image_path = list(annotations['file_name'].unique())
#        n_examples = len(image_path)
#
#        all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
#
#        for start, end in zip(range(0, n_examples, batch_size),
#                              range(batch_size, n_examples + batch_size, batch_size)):
#            image_batch_file = image_path[start:end]
#            image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
#                np.float32)
#            feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
#            all_feats[start:end, :] = feats
#            print ("Processed %d %s features.." % (end, split))
#        # use hickle to save huge feature vectors
#        hickle.dump(all_feats, save_path)
#        print ("Saved %s.." % (save_path))
        
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    data_path = './ai.challenger/image/test1_images_2017_resized/'
    save_path = './data/%s/%s.features.hkl' % ('test1', 'test1')
    pathDir = os.listdir(data_path)
    image_path = []
    
    n_examples = len(pathDir)
    for image in pathDir:
        image = os.path.join(data_path,image)
        image_path.append(image)
    save_pickle(image_path, './data/%s/%s.file.names.pkl' % ('test1', 'test1'))

    all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

    for start, end in zip(range(0, n_examples, batch_size),
                          range(batch_size, n_examples + batch_size, batch_size)):
        image_batch_file = image_path[start:end]
        image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
            np.float32)
        feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
        all_feats[start:end, :] = feats
        print ("Processed %d %s features.." % (end, 'test1'))
    # use hickle to save huge feature vectors
    hickle.dump(all_feats, save_path)
    print ("Saved %s.." % (save_path))
        
        
        
        
        
        