#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:23:50 2017

@author: root
"""
from core.utils import *
from core import captions2json
import json
import datetime

captions = load_pickle('./data/test1/test1.candidate.captions.pkl')
file_names = load_pickle('./data/test1/test1.file.names.pkl')
save_path = './data/test1/test1_cadidate_captions_json%d%d%d%d%d.json'%(  
            datetime.datetime.now().year,
            datetime.datetime.now().month,
            datetime.datetime.now().day, 
            datetime.datetime.now().hour, 
            datetime.datetime.now().minute)
                                                                
captions2json.captions2json(captions,file_names,save_path)

with open(save_path,'r') as f:
    
    test1_captions = json.load(f)