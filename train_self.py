
# coding: utf-8

# In[1]:


import json
import sys
import glob
import torch
sys.path.append('../')
import os
from transformers import *
from frameBERT.src import utils
from frameBERT.src import dataio
from frameBERT.src import eval_fn
from frameBERT import frame_parser
from frameBERT.src.modeling import BertForJointShallowSemanticParsing
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm, trange

from pprint import pprint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if device != "cpu":
    torch.cuda.set_device(0)
import pickle

import numpy as np
import random
np.random.seed(0)   
random.seed(0)

from torch import autograd
torch.cuda.empty_cache()

import argparse


# In[2]:


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


# # Define task

# In[3]:


srl = 'framenet'
language = 'multilingual'
fnversion = '1.2'

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, help='모델 폴더', default='/disk/frameBERT/cltl_eval/models/efn_ekfn_multitask/34')
parser.add_argument('--domain', required=True, help='도메인')
parser.add_argument('--result', required=False, help='결과 저장 폴더', default=False)
args = parser.parse_args()

print('#####')
print('\ttask:', srl)
print('\tlanguage:', language)
print('\tfn_version:', fnversion)
bert_io = utils.for_BERT(mode='train', language=language, masking=True, fnversion=fnversion)


# # Load data

# In[4]:


from koreanframenet import koreanframenet
kfn = koreanframenet.interface(version=fnversion)

en_trn, en_dev, en_tst = dataio.load_data(srl=srl, language='en')

ekfn_trn_d, ekfn_tst_d = kfn.load_data(source='efn')
jkfn_trn_d, jkfn_tst_d = kfn.load_data(source='jfn')
skfn_trn_d, skfn_unlabel_d, skfn_tst_d = kfn.load_data(source='sejong')
pkfn_trn_d, pkfn_unlabel_d, pkfn_tst_d = kfn.load_data(source='propbank')

ekfn_trn = dataio.data2tgt_data(ekfn_trn_d, mode='train')
ekfn_tst = dataio.data2tgt_data(ekfn_tst_d, mode='train')

jkfn_trn = dataio.data2tgt_data(jkfn_trn_d, mode='train')
jkfn_tst = dataio.data2tgt_data(jkfn_tst_d, mode='train')

skfn_trn = dataio.data2tgt_data(skfn_trn_d, mode='train')
skfn_unlabel = dataio.data2tgt_data(skfn_unlabel_d, mode='train')
skfn_tst = dataio.data2tgt_data(skfn_tst_d, mode='train')

pkfn_trn = dataio.data2tgt_data(pkfn_trn_d, mode='train')
pkfn_unlabel = dataio.data2tgt_data(pkfn_unlabel_d, mode='train')
pkfn_tst = dataio.data2tgt_data(pkfn_tst_d, mode='train')


# # Define Dataset

# In[5]:


trn_data = {}
trn_data['ekfn'] = ekfn_trn
trn_data['jkfn'] = jkfn_trn
trn_data['skfn'] = skfn_trn
trn_data['pkfn'] = pkfn_trn
trn_data['all'] = ekfn_trn + jkfn_trn + skfn_trn + pkfn_trn

tst_data = {}
tst_data['ekfn'] = ekfn_tst
tst_data['jkfn'] = jkfn_tst
tst_data['skfn'] = skfn_tst
tst_data['pkfn'] = pkfn_tst


unlabeled_data = {}
unlabeled_data['ekfn'] = ekfn_trn
unlabeled_data['jkfn'] = jkfn_trn
unlabeled_data['skfn'] = skfn_unlabel
unlabeled_data['pkfn'] = pkfn_unlabel
unlabeled_data['all'] = skfn_unlabel + pkfn_unlabel
# unlabeled_data['all'] = jkfn_trn + skfn_trn + skfn_unlabel + pkfn_trn + pkfn_unlabel
# unlabeled_data['skfn'] = skfn_trn + skfn_unlabel
# unlabeled_data['pkfn'] = pkfn_trn + pkfn_unlabel


# # Pre-trained Model

# In[6]:


pretrained_model = args.model

if args.model[-1] == '/':
    model_name = args.model.split('/')[-3]
else:
    model_name = args.model.split('/')[-2]
# pretrained_model = '/disk/frameBERT/models/enModel-fn17/2'
print('pretrained_model:', pretrained_model)


# # Parsing Unlabeld data

# In[7]:


def parsing_unlabeled_data(model_path, masking=True, language='ko', data='ekfn', threshold=0.7, added_list=[]):
#     torch.cuda.set_device(device)
    model = frame_parser.FrameParser(srl=srl,gold_pred=True, model_path=model_path, masking=masking, language=language, info=False)    
    result = []
    for i in range(len(unlabeled_data[data])):
        instance = unlabeled_data[data][i]
        
        if i not in added_list:

            parsed = model.parser(instance, result_format='all')        
            conll = parsed['conll'][0]
            frame_score = parsed['topk']['targets'][0]['frame_candidates'][0][-1]

            if frame_score >= float(threshold):
                parsed_result = conll
                result.append(parsed_result)
                added_list.append(i)
            
    added_list.sort()
        
    return result, added_list


# In[8]:


def train(model_path="bert-base-multilingual-cased",
          model_saved_path=False, epochs=3, batch_size=6, 
          trn=False): 
            
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    print('### START TRAINING:', model_saved_path)
    # load a pre-trained model first
    model = BertForJointShallowSemanticParsing.from_pretrained(model_path, 
                                                               num_senses = len(bert_io.sense2idx), 
                                                               num_args = len(bert_io.bio_arg2idx),
                                                               lufrmap=bert_io.lufrmap, 
                                                               frargmap = bert_io.bio_frargmap)
    model.to(device)
    
    print('\nconverting data to BERT input...')
    print('# of instances:', len(trn))
    trn_data = bert_io.convert_to_bert_input_JointShallowSemanticParsing(trn)
    sampler = RandomSampler(trn)
    trn_dataloader = DataLoader(trn_data, sampler=sampler, batch_size=batch_size)
    
    # load optimizer
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    
    max_grad_norm = 1.0
    
    for _ in trange(epochs, desc="Epoch"):
        
        # TRAIN loop
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            model.train()
            # add batch to gpu
            torch.cuda.set_device(device)
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_lus, b_input_senses, b_input_args, b_token_type_ids, b_input_masks = batch            
            loss = model(b_input_ids, lus=b_input_lus, senses=b_input_senses, args=b_input_args,
                     token_type_ids=b_token_type_ids, attention_mask=b_input_masks)
            
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            
            # update parameters
            optimizer.step()
            model.zero_grad()            

    # save your model at 10 epochs
    model.save_pretrained(model_saved_path)
    print('... TRAINNG is DONE')


# In[ ]:


model_saved_dir = '/disk/frameBERT/cltl_eval/models/'

if args.result:
    result_dir = args.result
else:
    result_dir = 'self_'+ args.domain +'_using_'+ model_name + '_with_labeled'
model_saved_dir = model_saved_dir + result_dir

if model_saved_dir[-1] != '/':
    model_saved_dir = model_saved_dir+'/'
    
if not os.path.exists(model_saved_dir):
    os.makedirs(model_saved_dir)
print('your models are saved to', model_saved_dir)
    
iters = 5
threshold = 0.9
instance = []
added_list = []
batch_size = 6

for _ in trange(iters, desc="Iteration"):
    iteration = _ + 1    
    
    if iteration == 1:
        pre_model = BertForJointShallowSemanticParsing.from_pretrained(pretrained_model, 
                                                               num_senses = len(bert_io.sense2idx), 
                                                               num_args = len(bert_io.bio_arg2idx),
                                                               lufrmap=bert_io.lufrmap, 
                                                               frargmap = bert_io.bio_frargmap)
        pre_model.to(device)
        
        model_saved_path = model_saved_dir+'0/'
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)
        pre_model.save_pretrained(model_saved_path)
        
    
        
        
    parsing_model_path = model_saved_dir + str(iteration-1) +'/'
    model_saved_path = model_saved_dir+str(iteration)+'/'
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    
    print('\n### ITERATION:', str(iteration))
    trn = trn_data['all']
    print('### PARSING START...')
    parsed_result, added_list = parsing_unlabeled_data(parsing_model_path, data=args.domain, 
                                                       masking=True, 
                                                       threshold=threshold, added_list=added_list)
    instance += parsed_result
    print('... is done')
    
    # training process
    trn_instance = trn + instance
    
    print('\n# of original training data:', len(trn))
    print('# of all unlabeled data:', len(unlabeled_data[args.domain]))
    print('# of psuedo labeled data:', len(instance), '('+str((round(len(instance)/len(unlabeled_data[args.domain])*100), 2))+'%)')
    print('Total Training Instance:', len(trn_instance), '\n') 
    
    train(model_path=parsing_model_path, model_saved_path=model_saved_path, trn=trn_instance)    


# # Training
