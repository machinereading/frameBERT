
# coding: utf-8

# In[1]:


import json
import os
import frame_parser
from src import dataio
import glob
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score
import random

import torch
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if device != "cpu":
    torch.cuda.set_device(0)


# In[2]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


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


# In[5]:


def test(srl=False, masking=False, viterbi=False, language=False, model_path=False, 
         result_dir=False, train_lang=False, tgt=False,
         pretrained="bert-base-multilingual-cased"):
    if not result_dir:
        result_dir = '/disk/data/models/'+model_dir.split('/')[-2]+'-result/'
    else:
        pass
    if result_dir[-1] != '/':
        result_dir = result_dir+'/'
        
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    if not train_lang:
        train_lang = language
    
    fname = result_dir+train_lang+'_for_'+language
        
    if masking:
        fname = fname + '_with_masking_result.txt'
    else:
        fname = fname +'_result.txt'
        
    print('### Your result would be saved to:', fname)
        
    trn, dev, tst = dataio.load_data(srl=srl, language=language, exem=False, info=False)
    if srl == 'framenet-argid':
        trn = dataio.fe2arg(trn)
        dev = dataio.fe2arg(dev)
        tst = dataio.fe2arg(tst)
    else:
        pass
    
    print('### EVALUATION')
    print('MODE:', srl)
    print('target LANGUAGE:', language)
    print('trained LANGUAGE:', train_lang)
    print('Viterbi:', viterbi)
    print('masking:', masking)
    print('using TGT token:', tgt)
    tic()    
    
    models = glob.glob(model_path+'*/')
    
    eval_result = []
    for m in models:        
        print('### model dir:', m)
        print('### TARGET LANGUAGE:', language)
        torch.cuda.set_device(device)
        
        
        model = frame_parser.FrameParser(srl=srl,gold_pred=True, model_path=m, viterbi=viterbi, 
                                         masking=masking, language=language, tgt=tgt,
                                         pretrained=pretrained)

        gold_senses, pred_senses, gold_args, pred_args = [],[],[],[]        
        gold_full_all, pred_full_all = [],[]

        for instance in tst:
            torch.cuda.set_device(device)
            result = model.parser(instance)

            gold_sense = [i for i in instance[2] if i != '_'][0]
            pred_sense = [i for i in result[0][2] if i != '_'][0]


            gold_arg = [i for i in instance[3] if i != 'X']
            pred_arg = [i for i in result[0][3]]

            gold_senses.append(gold_sense)
            pred_senses.append(pred_sense)
            
            weighted_gold_frame, weighted_pred_frame, weighted_gold_arg, weighted_pred_arg = weighting(gold_sense, pred_sense, gold_arg, pred_arg)
            
            gold_args.append(weighted_gold_arg)
            pred_args.append(weighted_pred_arg)

#             if 'framenet' in srl:
            gold_full = []
            gold_full += weighted_gold_frame
            gold_full += weighted_gold_arg

            pred_full = []
            pred_full += weighted_pred_frame
            pred_full += weighted_pred_arg

            gold_full_all.append(gold_full)
            pred_full_all.append(pred_full)
                
#             break
                
        del model
            
        acc = accuracy_score(gold_senses, pred_senses)
        arg_f1 = f1_score(gold_args, pred_args)
        arg_precision = precision_score(gold_args, pred_args)
        arg_recall = recall_score(gold_args, pred_args)

        epoch = m.split('/')[-2]
        print('# EPOCH:', epoch)
        print("SenseId Accuracy: {}".format(acc))
        print("ArgId Precision: {}".format(arg_precision))
        print("ArgId Recall: {}".format(arg_recall))
        print("ArgId F1: {}".format(arg_f1))
#         if 'framenet' in srl:
        full_f1 = f1_score(gold_full_all, pred_full_all)
        full_precision = precision_score(gold_full_all, pred_full_all)
        full_recall = recall_score(gold_full_all, pred_full_all)
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
        
#         break
    
    with open(fname,'w') as f:
        f.write('epoch'+'\t''SenseID'+'\t'+'Arg_P'+'\t'+'Arg_R'+'\t'+'ArgF1'+'\t'+'full_P'+'\t'+'full_R'+'\t'+'full_F1'+'\n')
        for i in eval_result:
            line = '\t'.join(i)
            f.write(line+'\n')
            
        print('\n\t### Your result is saved at:', fname)
        print('...done\n')


# In[6]:


srl = 'framenet'


# In[1]:


#평가 1-1. ArgID 모델 평가 for En

# language = 'en'

# print('\n####################################################')
# print('\t###eval for ARGID Model (masking) for English')
# model_path = '/disk/frameBERT/models/en_argid/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl='framenet-argid', language=language, masking=True, viterbi=False, tgt=True, train_lang='en_argid', 
#      model_path=model_path, result_dir=result_dir)


# In[2]:


# 평가 1-2. ArgID_only 모델 평가 for En

# language = 'en'

# print('\n####################################################')
# print('\t###eval for ARGID_only Model (masking) for English')
# model_path = '/disk/frameBERT/models/en_argid_only_arg/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl='framenet-argid', language=language, masking=True, viterbi=False, tgt=True, train_lang='en_argid_only_arg', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 1-3. ArgID 모델 평가 for Ko

# language = 'ko'

# print('\n####################################################')
# print('\t###eval for ARGID Model (masking) for Korean')
# model_path = '/disk/frameBERT/models/ko_argid/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl='framenet-argid', language=language, masking=True, viterbi=False, tgt=True, train_lang='ko_argid', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 1-4. ArgID_only 모델 평가 for Ko

# language = 'ko'

# print('\n####################################################')
# print('\t###eval for ARGID_only Model (masking) for Korean')
# model_path = '/disk/frameBERT/models/ko_argid_only_arg/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl='framenet-argid', language=language, masking=True, viterbi=False, tgt=True, train_lang='ko_argid_only_arg', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 2. En 모델을 En에 평가

# language = 'en'

# print('\n####################################################')
# print('\t###eval for en Model (masking) for English')
# model_path = '/disk/frameBERT/models/enModel-with-exemplar/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='en', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 3. Ko 모델을 Ko에 평가

# language = 'ko'

# print('\t###eval for ko Model (masking) for Korean')
# model_path = '/disk/frameBERT/models/koModel/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='ko', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 4. Cross-lingual 모델을 Ko에 평가

# language = 'ko'

# print('\n####################################################')
# print('\t###eval for Cross-lingual Model (masking) for Korean')
# model_path = '/disk/frameBERT/models/crosslingual/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='cross', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 5. En 모델을 Ko에 평가 (Zero-shot)

# language = 'ko'

# print('\n####################################################')
# print('\t###eval for En Model (masking) for Korean')
# model_path = '/disk/frameBERT/models/enModel-with-exemplar/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='zero-shot', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 6. Joint 모델을 Ko에 평가

# language = 'ko'

# print('\n####################################################')
# print('\t###eval for Joint Model (masking) for Korean')
# model_path = '/disk/frameBERT/models/joint/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='multilingual-100', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 7. Joint 모델을 En에 평가

language = 'en'

print('\n####################################################')
print('\t###eval for Joint Model (masking) for English')
model_path = '/disk/frameBERT/models/joint/'
result_dir = '/disk/frameBERT/eval_result/'

test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='multilingual-100', 
     model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 8. Joint 모델 - 10%, 25%, 50%, 75%

# language = 'ko'

# print('\n####################################################')
# print('\t###eval for Joint Model (masking) for Korean by increasing dataset')
# model_path = '/disk/data/models/increasing_data/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='multilingual-increasing', 
#      model_path=model_path, result_dir=result_dir)


# In[ ]:


# 평가 9. ko 모델 - 10%, 25%, 50%, 75%

# language = 'ko'

# print('\n####################################################')
# print('\t###eval for 10% Joint Model (masking) for Korean by increasing dataset')
# model_path = '/disk/data/models/increasing_data_only_ko/'
# result_dir = '/disk/frameBERT/eval_result/'

# test(srl=srl, language=language, masking=True, viterbi=False, tgt=True, train_lang='ko-increasing', 
#      model_path=model_path, result_dir=result_dir)

