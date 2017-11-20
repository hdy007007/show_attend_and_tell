#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 00:42:11 2017

@author: root
"""

import json
import pickle
import hashlib
import jieba

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def captions2json(captions,file_names,save_path):
    file_to_captions = {}
    for c,f in zip(captions,file_names):
        c = c.replace(' ','')
        f = f.split('/')[-1].split('.')[0]
        file_to_captions[f] =  c
    with open(save_path,'w') as f:
        json.dump(file_to_captions,f)

with open('/home/houdanyang/tensorflow/show-attend-and-tell/data/val/val.file.names.pkl') as f:
    file_names = pickle.load(f)


    
with open('/home/houdanyang/tensorflow/show-attend-and-tell/data/val/val.references.pkl') as f:
    captions = pickle.load(f)

val_dict = {}
val_dict['type'] ='captions'
val_dict['annotations'] = []
val_dict['images'] = []
i = 1
for c in captions:
    
    file_names_t = file_names[int(c)].split('/')[-1].split('.')[0]
    file_names_hash = int(int(hashlib.sha256(file_names_t).hexdigest(), 16) % sys.maxint)
    image_json_t = {}
    image_json_t['file_name'] =file_names_t
    image_json_t['id'] = file_names_hash
    
    for n in captions[c]:
        val_dict['images'].append(image_json_t)
        temp = n.replace(".","")
        words = jieba.cut(temp)
        caption_t = " ".join(words)
        t_json = {}
        t_json['id'] = i
        t_json['caption'] = caption_t
        t_json['image_id'] = file_names_hash
        val_dict['annotations'].append(t_json)      
        i = i + 1
        
val_dict['info'] = {
    "contributor": "He Zheng",
    "description": "CaptionEval",
    "url": "https://github.com/AIChallenger/AI_Challenger.git",
    "version": "1",
    "year": 2017
  }
val_dict['licenses'] = {'url':"https//challenger.ai"} 
with open('/home/houdanyang/tensorflow/show-attend-and-tell/AI_Challenger/Evaluation/caption_eval/data/val_references_json.json','w') as f:
    json.dump(val_dict,f,ensure_ascii=False)    