#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:35:47 2017

@author: root
"""

import json
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def captions2json(captions,file_names,save_path):
    file_to_captions = []
    for c,f in zip(captions,file_names):
        c = c.replace(' ','')
        f = f.split('/')[-1].split('.')[0]
        json_t = {}
        json_t['caption'] = c
        json_t['image_id'] = f
        file_to_captions.append(json_t)

    with open(save_path,'w') as f:
        json.dump(file_to_captions,f,ensure_ascii=False)    

#with open('/home/houdanyang/tensorflow/show-attend-and-tell/data/val/val.file.names.pkl') as f:
#    file_names = pickle.load(f)
#
#with open('/home/houdanyang/tensorflow/show-attend-and-tell/data/val/val.candidate.captions.pkl') as f:
#    captions = pickle.load(f)
#
#captions2json(captions,file_names,'/home/houdanyang/tensorflow/show-attend-and-tell/data/val/val_cadidate_captions_json.json')
