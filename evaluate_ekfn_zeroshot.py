
# coding: utf-8

# In[1]:


import json
import os
import frame_parser
from src import dataio
from src import eval_fn
import glob
from pprint import pprint


# # Define task

# In[ ]:


# model_path = '/disk/frameBERT/models/enModel-with-exemplar/'
model_path = '/disk/frameBERT/models/enModel-fn17/'
fname = '/disk/frameBERT/cltl_eval/eval_result/ekfn_zeroshot_wo_exem.txt'


# In[2]:


srl = 'framenet'
language = 'ko'
fnversion = '1.2'


# # Load data

# In[3]:


from koreanframenet import koreanframenet
kfn = koreanframenet.interface(version=fnversion)

ekfn_trn_d = kfn.load_data(source='efn')
ekfn_tst_d = kfn.load_data(source='efn_test')
jkfn_d = kfn.load_data(source='jfn')
skfn_d = kfn.load_data(source='sejong')
pkfn_d = kfn.load_data(source='propbank')

ekfn_trn = dataio.data2tgt_data(ekfn_trn_d, mode='train')
ekfn_tst = dataio.data2tgt_data(ekfn_tst_d, mode='train')
jkfn = dataio.data2tgt_data(jkfn_d, mode='train')
skfn = dataio.data2tgt_data(skfn_d, mode='train')
pkfn = dataio.data2tgt_data(pkfn_d, mode='train')


# In[ ]:


tst = ekfn_tst
print(len(tst))
print(tst[0])


# # Evaluate Models

# In[ ]:


# Parsing Gold Data

def test_model(model_path, masking=True, language='ko'):
#     torch.cuda.set_device(device)
    model = frame_parser.FrameParser(srl=srl,gold_pred=True, 
                                     fnversion=fnversion,
                                     model_path=model_path, 
                                     masking=masking, 
                                     language=language)
    
    parsed_result = []
    for instance in tst:
#         torch.cuda.set_device(device)
        result = model.parser(instance)[0]
        parsed_result.append(result)
        
#         break
        
    return parsed_result


# In[6]:


models = glob.glob(model_path+'*')

result = {}

for model_path in models:
    print('model:', model_path)
    parsed_result = test_model(model_path, language=language)
    frameid, arg_precision, arg_recall, arg_f1, full_precision, full_recall, full_f1 = eval_fn.evaluate(tst, parsed_result)
    
    d = {}
    d['frameid'] = frameid
    d['arg_precision'] = arg_precision
    d['arg_recall'] = arg_recall
    d['arg_f1'] = arg_f1
    d['full_precision'] = full_precision
    d['full_recall'] = full_recall
    d['full_f1'] = full_f1
    result[model_path] = d
    pprint(d)


# In[7]:


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

with open(fname, 'w') as f:
    for line in lines:
        f.write(line + '\n')
        
print('######')
print('\tlanguage:', language)
print('\tfnversion:', fnversion)
print('\teval result:')
pprint(result)
print('\n....is written at', fname)

