
# coding: utf-8

# In[1]:


import json
import os
import frame_parser
from src import dataio
from src import eval_fn
import glob
from pprint import pprint

import argparse


# # Define task

# In[2]:


srl = 'framenet'
parser = argparse.ArgumentParser()
parser.add_argument('--language', required=False, help='choose target language', default='ko')
parser.add_argument('--model', required=False, help='모델 폴더', default='/disk/frameBERT/cltl_eval/models/ekfn/')
parser.add_argument('--eval_model', required=False, help='여러개 모델인지', default='all')
parser.add_argument('--test', required=True, help='테스트 데이터')
parser.add_argument('--result', required=False, help='결과 저장 폴더', default='/disk/frameBERT/cltl_eval/eval_result/')
args = parser.parse_args()

if args.model[-1] != '/':
    args.model = args.model+'/'
model_name = args.model[:-1]

if args.result[-1] == '/':
    args.result = args.result[:-1]

if model_name.split('/')[-1] == 'best':
    model_name = model_name.split('/')[-2]
else:
    model_name = model_name.split('/')[-1]
result_fname = args.result+'/'+model_name+'_to_'+args.test+'.txt'

print('##### model:', args.model)
print('##### test data:', args.test)
print('##### your result file:', result_fname)


# # Load data

# In[3]:


from koreanframenet import koreanframenet
kfn = koreanframenet.interface(version='1.2')

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


# In[ ]:


tst = {}
tst['ekfn'] = ekfn_tst
tst['jkfn'] = jkfn_tst
tst['skfn'] = skfn_tst
tst['pkfn'] = pkfn_tst

print('test data:', args.test)


# In[5]:


# Parsing Gold Data

def test_model(model_path, masking=True, language='en', test_data='ekfn'):
#     torch.cuda.set_device(device)
    model = frame_parser.FrameParser(srl=srl,gold_pred=True, model_path=model_path, masking=masking, language=language)
    
    parsed_result = []
    for instance in tst[test_data]:
#         torch.cuda.set_device(device)
        result = model.parser(instance)[0]
        parsed_result.append(result)        
#         break
        
    return parsed_result
        
# parsed = test_model('/disk/frameBERT/models/joint/36/', language=language)


# # Evaluate Models

# In[7]:


# model_path = '/disk/frameBERT/models/crosslingual/'
# models = args.model
if args.eval_model == 'all':
    models = glob.glob(args.model+'*')
else:
    models = [args.model]

result = {}

for model_path in models:
    parsed_result = test_model(model_path, language=args.language, test_data=args.test)
    frameid, arg_precision, arg_recall, arg_f1, full_precision, full_recall, full_f1 = eval_fn.evaluate(tst[args.test], parsed_result)
    
    d = {}
    d['frameid'] = frameid
    d['arg_precision'] = arg_precision
    d['arg_recall'] = arg_recall
    d['arg_f1'] = arg_f1
    d['full_precision'] = full_precision
    d['full_recall'] = full_recall
    d['full_f1'] = full_f1
    result[model_path] = d
    print('model:', model_path)
    print('test:', args.test)
    pprint(d)
#     break
    
pprint(result)


# In[8]:


# write file as csv format

lines = []
lines.append('epoch'+'\t'+'SenseID'+'\t'+'Arg_P'+'\t'+'Arg_R'+'\t'+'ArgF1'+'\t'+'full_P'+'\t'+'full_R'+'\t'+'full_F1')
for m in result:
    epoch = m.split('/')[-1]
    senseid = str(result[m]['frameid'])
    arg_p = str(result[m]['arg_precision'])
    arg_r = str(result[m]['arg_recall'])
    arg_f1 = str(result[m]['arg_f1'])
    full_p = str(result[m]['full_precision'])
    full_r = str(result[m]['full_recall'])
    full_f1 = str(result[m]['full_f1'])
    line = epoch+'\t'+senseid+'\t'+arg_p+'\t'+arg_r+'\t'+arg_f1+'\t'+full_p+'\t'+full_r+'\t'+full_f1
    lines.append(line)    

    
with open(result_fname, 'w') as f:
    for line in lines:
        f.write(line + '\n')
        
print('######')
print('\teval result is written at', result_fname)

