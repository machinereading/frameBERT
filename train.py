
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
from frameBERT import frame_parser
from frameBERT.src.modeling import BertForJointShallowSemanticParsing
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score

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

from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score


# In[2]:


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


# In[3]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[4]:


def train(trn=False, PRETRAINED_MODEL="bert-base-multilingual-cased",
          model_dir=False, epochs=20, fnversion=False):
    print('\tyour model would be saved at', model_dir)

    # load a model first
    model = BertForJointShallowSemanticParsing.from_pretrained(PRETRAINED_MODEL, 
                                                               num_senses = len(bert_io.sense2idx), 
                                                               num_args = len(bert_io.bio_arg2idx),
                                                               lufrmap=bert_io.lufrmap, 
                                                               frargmap = bert_io.bio_frargmap)
    model.to(device)
    
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
    num_of_epoch = 0
    accuracy_result = []    
    best_score = 0
    
    early_stopping = False
    early_stopping = True
    renew_stack = 0
    
    for _ in trange(epochs, desc="Epoch"):
        
        # TRAIN loop
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            model.train()
            # add batch to gpu
            torch.cuda.set_device(device)
#             torch.cuda.set_device(0)
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
    
#             break

        # save your model
        model_saved_path = model_dir+str(num_of_epoch)+'/'
        print('\n\tyour model is saved:', model_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)
        model.save_pretrained(model_saved_path)

        num_of_epoch += 1
        
    print('...training is done\n')


# In[5]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

with open('./data/frame_coreFE_list.json','r') as f:
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


# In[14]:


def test(masking=True, model_path=False, tst_language='ko', trn_language='ko',
         pretrained="bert-base-multilingual-cased", mode='tst', fnversion=False, 
         result_dir=False):
        
    trn, dev, tst = dataio.load_data(srl=srl, language=tst_language, exem=False, info=False, fnversion=fnversion)
    
    
    if mode == 'tst':
        data = tst
    else:
        data = dev
        
    models = glob.glob(model_path+'*/')
    eval_result = []
    
    fname = result_dir+'kfn'+str(fnversion)+'.txt'

    tic()
    for m in models:
        print('### model dir:', m)
        torch.cuda.set_device(device) 
        tst_model = frame_parser.FrameParser(gold_pred=True, model_path=m, viterbi=False, 
                                         fnversion=fnversion, masking=masking, language=trn_language, tgt=True,
                                         pretrained="bert-base-multilingual-cased", info=True)
        
        
        gold_senses, pred_senses, gold_args, pred_args = [],[],[],[]        
        gold_full_all, pred_full_all = [],[]        
        
        for instance in data:
            torch.cuda.set_device(device)
            result = tst_model.parser(instance)

            gold_sense = [i for i in instance[2] if i != '_'][0]
            pred_sense = [i for i in result[0][2] if i != '_'][0]

            gold_arg = [i for i in instance[3] if i != 'X']
            pred_arg = [i for i in result[0][3]]

            gold_senses.append(gold_sense)
            pred_senses.append(pred_sense)

            weighted_gold_frame, weighted_pred_frame, weighted_gold_arg, weighted_pred_arg = weighting(gold_sense, pred_sense, gold_arg, pred_arg)

            gold_args.append(weighted_gold_arg)
            pred_args.append(weighted_pred_arg)            

            gold_full = []
            gold_full += weighted_gold_frame
            gold_full += weighted_gold_arg

            pred_full = []
            pred_full += weighted_pred_frame
            pred_full += weighted_pred_arg

            gold_full_all.append(gold_full)
            pred_full_all.append(pred_full)

#             break

        del tst_model

        acc = accuracy_score(gold_senses, pred_senses)
        arg_f1 = f1_score(gold_args, pred_args)
        arg_precision = precision_score(gold_args, pred_args)
        arg_recall = recall_score(gold_args, pred_args)
        full_f1 = f1_score(gold_full_all, pred_full_all)
        full_precision = precision_score(gold_full_all, pred_full_all)
        full_recall = recall_score(gold_full_all, pred_full_all)
        
        epoch = m.split('/')[-2]
        print('# EPOCH:', epoch)
        print("SenseId Accuracy: {}".format(acc))
        print("ArgId Precision: {}".format(arg_precision))
        print("ArgId Recall: {}".format(arg_recall))
        print("ArgId F1: {}".format(arg_f1))
        print("full-structure Precision: {}".format(full_precision))
        print("full-structure Recall: {}".format(full_recall))
        print("full-structure F1: {}".format(full_f1))
        print('-----processing time:', tac())
        print('')
        
        model_result = []
        model_result.append(epoch)
        model_result.append(acc)
        model_result.append(arg_precision)
        model_result.append(arg_recall)
        model_result.append(arg_f1)
#         if 'framenet' in srl:
        model_result.append(full_precision)
        model_result.append(full_recall)
        model_result.append(full_f1)
        model_result = [str(i) for i in model_result]
        eval_result.append(model_result)
        
    with open(fname,'w') as f:
        f.write('epoch'+'\t''SenseID'+'\t'+'Arg_P'+'\t'+'Arg_R'+'\t'+'ArgF1'+'\t'+'full_P'+'\t'+'full_R'+'\t'+'full_F1'+'\n')
        for i in eval_result:
            line = '\t'.join(i)
            f.write(line+'\n')
            
        print('\n\t### Your result is saved at:', fname)
        print('...done\n')


# In[7]:


srl = 'framenet'
epochs = 20
masking = True
MAX_LEN = 256
batch_size = 6
PRETRAINED_MODEL = "bert-base-multilingual-cased"
print('')
print('### TRAINING')
print('MODEL:', srl)
print('PRETRAINED BERT:', PRETRAINED_MODEL)
print('BATCH_SIZE:', batch_size)
print('MAX_LEN:', MAX_LEN)
print('epochs:', epochs)
print('')


# In[8]:


# model_dir = '/disk/frameBERT/model-kfn08/'
# fnversion = '0.8'
# language = 'ko'
# bert_io = utils.for_BERT(mode='train', language=language, masking=True, fnversion=fnversion)
# trn, dev, tst = dataio.load_data(srl=srl, language=language, fnversion=fnversion)

# train(trn=trn, epochs=epochs, model_dir=model_dir, fnversion=fnversion)


# In[9]:


# model_dir = '/disk/frameBERT/model-kfn10/'
# fnversion = '1.0'
# language = 'ko'
# bert_io = utils.for_BERT(mode='train', language=language, masking=True, fnversion=fnversion)
# trn, dev, tst = dataio.load_data(srl=srl, language=language, fnversion=fnversion)

# train(trn=trn, epochs=epochs, model_dir=model_dir, fnversion=fnversion)


# In[15]:


# model_path = '/disk/frameBERT/model-kfn08/'
# result_dir = '/disk/frameBERT/eval_result/'
# fnversion = '0.8'

# test(model_path=model_path, result_dir=result_dir, fnversion=fnversion)


# In[ ]:


model_path = '/disk/frameBERT/model-kfn10/'
result_dir = '/disk/frameBERT/eval_result/'
fnversion = '1.0'

test(model_path=model_path, result_dir=result_dir, fnversion=fnversion)

