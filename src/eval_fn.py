import json
import os
import frame_parser
from src import dataio
import glob
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score
import random

import torch

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'
    
# 실행시간 측정 함수
import time

_start_time = time.time()

def tic():
    global _start_time 
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    
    result = '{}hour:{}min:{}sec'.format(t_hour,t_min,t_sec)
    return result


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

with open(dir_path+'/../data/frame_coreFE_list.json','r') as f:
    frame_coreFE = json.load(f)

def weighting(gold_frame, pred_frame, gold_args, pred_args):
    
    weighted_gold_frame, weighted_pred_frame = ['B-'+gold_frame], ['B-'+pred_frame]
    weighted_gold_args, weighted_pred_args = gold_args.copy(), pred_args.copy()
      
    weighted_gold_frame.append('B-'+gold_frame)
    weighted_pred_frame.append('B-'+pred_frame)        

    for i in range(len(gold_args)):
        gold_arg = gold_args[i]
        pred_arg = pred_args[i]

        fe = gold_arg.split('-')[-1]
        if fe in frame_coreFE[gold_frame]:
            weighted_gold_args.append(gold_arg)
            weighted_pred_args.append(pred_arg)
        elif fe == 'ARG':
            weighted_gold_args.append(gold_arg)
            weighted_pred_args.append(pred_arg)
        else:
            weighted_gold_args.append('O')
            weighted_pred_args.append('O')
        
    return weighted_gold_frame, weighted_pred_frame, weighted_gold_args, weighted_pred_args


def evaluate(gold_data, parsed_data):
    
    tic()
    
    gold_frames, pred_frames = [],[]
    gold_args, pred_args = [],[]
    gold_fulls, pred_fulls = [],[]
    
    for i in range(len(parsed_data)):
        gold = gold_data[i]
        parsed = parsed_data[i]
        
        gold_frame = [i for i in gold[2] if i != '_'][0]
        pred_frame = [i for i in parsed[2] if i != '_'][0]
        
        gold_arg = [i for i in gold[3] if i != 'X']
        pred_arg = [i for i in parsed[3]]
        
        gold_frames.append(gold_frame)
        pred_frames.append(pred_frame)
        
        weighted_gold_frame, weighted_pred_frame, weighted_gold_arg, weighted_pred_arg = weighting(gold_frame, pred_frame, gold_arg, pred_arg)
        
        gold_args.append(weighted_gold_arg)
        pred_args.append(weighted_pred_arg)
        
        gold_full = []
        gold_full += weighted_gold_frame
        gold_full += weighted_gold_arg

        pred_full = []
        pred_full += weighted_pred_frame
        pred_full += weighted_pred_arg

        gold_fulls.append(gold_full)
        pred_fulls.append(pred_full)
        
        
            
    acc = accuracy_score(gold_frames, pred_frames)
    arg_f1 = f1_score(gold_args, pred_args)
    arg_precision = precision_score(gold_args, pred_args)
    arg_recall = recall_score(gold_args, pred_args)

    full_f1 = f1_score(gold_fulls, pred_fulls)
    full_precision = precision_score(gold_fulls, pred_fulls)
    full_recall = recall_score(gold_fulls, pred_fulls)
    
    result = (acc, arg_precision, arg_recall, arg_f1, full_precision, full_recall, full_f1)
    
    print('evaluation is complete:',tac())
    
    return result




