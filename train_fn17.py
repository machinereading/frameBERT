
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


# # Define task

# In[2]:


srl = 'framenet'
language = 'en'
fnversion = '1.7'

print('#####')
print('\ttask:', srl)
print('\tlanguage:', language)
print('\tfn_version:', fnversion)


# In[3]:


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


# In[4]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[5]:


def train(PRETRAINED_MODEL="bert-base-multilingual-cased",
          model_dir=False, epochs=20, fnversion=False, early_stopping=True, batch_size=6):
    
    tic()
    
    if model_dir[-1] != '/':
        model_dir = model_dir+'/'
        
    if early_stopping == True:
        model_saved_path = model_dir+'best/'
        model_dummy_path = model_dir+'dummy/'
        if not os.path.exists(model_dummy_path):
            os.makedirs(model_dummy_path)
    else:
        model_saved_path = model_dir        
            
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    print('\nyour model would be saved at', model_saved_path)

    # load a pre-trained model first
    print('\nloading a pre-trained model...')
    model = BertForJointShallowSemanticParsing.from_pretrained(PRETRAINED_MODEL, 
                                                               num_senses = len(bert_io.sense2idx), 
                                                               num_args = len(bert_io.bio_arg2idx),
                                                               lufrmap=bert_io.lufrmap, 
                                                               frargmap = bert_io.bio_frargmap)
    model.to(device)
    print('... is done.', tac())
    
    print('\nconverting data to BERT input...')
    trn_data = bert_io.convert_to_bert_input_JointShallowSemanticParsing(trn)
    sampler = RandomSampler(trn)
    trn_dataloader = DataLoader(trn_data, sampler=sampler, batch_size=batch_size)
    print('... is done', tac())
    
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
    num_of_epoch = 0
    
    best_score = 0
    renew_stack = 0
    
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

        if early_stopping == True:
            model.save_pretrained(model_dummy_path)
            
            # evaluate the model using dev dataset
            print('\n### eval by dev')
            test_model = frame_parser.FrameParser(srl=srl, gold_pred=True, 
                                                  info=False, model_path=model_dummy_path, language=language)
            parsed_result = []
            
            for instance in dev:
                result = test_model.parser(instance)[0]
                parsed_result.append(result)
                
            del test_model
                
            frameid, arg_precision, arg_recall, arg_f1, full_precision, full_recall, full_f1 = eval_fn.evaluate(dev, parsed_result)
            d = {}
            d['frameid'] = frameid
            d['arg_precision'] = arg_precision
            d['arg_recall'] = arg_recall
            d['arg_f1'] = arg_f1
            d['full_precision'] = full_precision
            d['full_recall'] = full_recall
            d['full_f1'] = full_f1
            pprint(d)
            print('Best score:', best_score)
            
            if full_f1 > best_score:
                model.save_pretrained(model_saved_path)
                best_score = full_f1
                
                renew_stack = 0
            else:
                renew_stack +=1
        
            # 성능이 3epoch 이후에도 개선되지 않으면 중단
            if renew_stack >= 3:
                break
            
        elif early_stopping == False:
            # save your model for each epochs
            model_saved_path = model_dir+str(num_of_epoch)+'/'
            if not os.path.exists(model_saved_path):
                os.makedirs(model_saved_path)
            model.save_pretrained(model_saved_path)

            num_of_epoch += 1
            
        
    print('...training is done. (', tac(), ')')


# # Load dataset

# In[6]:


bert_io = utils.for_BERT(mode='train', language=language, masking=True, fnversion=fnversion)
trn, dev, tst = dataio.load_data(language=language, fnversion=fnversion)
print(trn[0])


# # Training

# In[8]:


epochs = 20
model_dir = '/disk/frameBERT/models/enModel-fn17'
early_stopping = False
batch_size = 6

train(epochs=epochs, model_dir=model_dir, fnversion=fnversion, early_stopping=early_stopping, batch_size=batch_size)

