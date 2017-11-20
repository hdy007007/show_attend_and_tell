 
# encoding: utf-8
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The unittest for image Chinese captioning evaluation."""
# __author__ = 'ZhengHe'
# python2.7
# python run_evaluations.py

import sys


reload(sys)
sys.setdefaultencoding('utf8')
from run_evaluations import compute_m1
import json

m1_score = compute_m1(json_predictions_file="./data/val_cadidate_captions_json.json",
                              reference_file="./data/val_references_json.json")

#with open('/home/houdanyang/tensorflow/show-attend-and-tell/AI_Challenger/Evaluation/caption_eval/data/val_captions_json.json') as f:
#    val_captions = json.load(f)
#
#with open('/home/houdanyang/tensorflow/show-attend-and-tell/AI_Challenger/Evaluation/caption_eval/data/val_references_json.json') as f:
#    val_references = json.load(f)    
#   
#with open('/home/houdanyang/tensorflow/show-attend-and-tell/AI_Challenger/Evaluation/caption_eval/data/id_to_words.json') as f:
#    id_to_words = json.load(f)   
#
#with open('/home/houdanyang/tensorflow/show-attend-and-tell/AI_Challenger/Evaluation/caption_eval/data/id_to_test_caption.json') as f:
#    id_to_test_words = json.load(f)   
#
#val_captions = val_captions[:3]
#val_references['annotations'] = val_references['annotations'][:14]
#val_references['images'] = val_references['images'][:14]
###
#with open('/home/houdanyang/tensorflow/show-attend-and-tell/AI_Challenger/Evaluation/caption_eval/data/val_captions_json1.json','w') as f:
#    json.dump(val_captions,f,ensure_ascii=False)  
#
#with open('/home/houdanyang/tensorflow/show-attend-and-tell/AI_Challenger/Evaluation/caption_eval/data/val_references_json1.json','w') as f:
#    json.dump(val_references,f,ensure_ascii=False)  