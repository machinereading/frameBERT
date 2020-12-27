
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

# In[2]:


srl = 'framenet'
language = 'en'
fnversion = 1.7


# # Load data

# In[3]:


trn, dev, tst = dataio.load_data(srl=srl, language=language, fnversion=fnversion, exem=False, info=True)


# In[4]:


# Parsing Gold Data

def test_model(model_path, masking=True, language='en'):
#     torch.cuda.set_device(device)
    model = frame_parser.FrameParser(srl=srl,gold_pred=True, fnversion=fnversion,
                                     model_path=model_path, masking=masking, language=language)
    
    parsed_result = []
    for instance in tst:
#         torch.cuda.set_device(device)
        result = model.parser(instance)[0]
        parsed_result.append(result)
        
#         break
        
    return parsed_result
        
# parsed = test_model('/disk/frameBERT/models/joint/36/', language=language)


# # Data format example

# In[5]:


print('\ntest_data')
print(tst[0])

# print('\nparsed_data')
# print(parsed[0])


# # Evaluate Models

# In[6]:


model_path = '/disk/frameBERT/models/argid-fn17-exem/'
models = glob.glob(model_path+'*')

result = {}

for model_path in models:
    print('model:', model_path)
    parsed_result = test_model(model_path, language=language)
    frameid, arg_precision, arg_recall, arg_f1, full_precision, full_recall, full_f1 = eval_fn.evaluate(tst, parsed_result, 
                                                                                                        fnversion=fnversion)
    
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
    
#     break
    
# pprint(result)


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
    

fname = '/disk/frameBERT/eval_result/argid-fn17-exem.txt'
with open(fname, 'w') as f:
    for line in lines:
        f.write(line + '\n')
        
print('######')
print('\tlanguage:', language)
print('\tfnversion:', fnversion)
print('\teval result:')
pprint(result)
print('\n....is written at', fname)

