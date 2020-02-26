import json
import sys
import glob
import torch
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from frameBERT.koreanframenet import koreanframenet
from frameBERT.src import kotimex

from konlpy.tag import Kkma
kkma = Kkma()
import jpype
jpype.attachThreadToJVM()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'
    
dir_path = dir_path+'/..'
    
def conll2tagseq(data):
    tokens, preds, senses, args = [],[],[],[]
    result = []
    for line in data:
        line = line.strip()
        if line.startswith('#'):
            pass
        elif line != '':
            t = line.split('\t')
            token, pred, sense, arg = t[1], t[2], t[3], t[4]            
            tokens.append(token)
            preds.append(pred)
            senses.append(sense)
            args.append(arg)
        else:            
            sent = []
            sent.append(tokens)
            sent.append(preds)
            sent.append(senses)
            sent.append(args)            
            result.append(sent)
            tokens, preds, senses, args = [],[],[],[]
            
    return result
    
def load_data(srl='framenet', language='ko', fnversion=1.1, path=False, exem=True):
    if srl == 'framenet':
        if language == 'ko':
            kfn = koreanframenet.interface(version=fnversion)
            trn_d, dev_d, tst_d = kfn.load_data()
        else:
            if path == False:
                fn_dir = '/disk/FrameNet/resource/fn1.7/'
            else:
                fn_dir = path
            with open(fn_dir+'fn1.7.fulltext.train.syntaxnet.conll') as f:
                d = f.readlines()
            trn_d = conll2tagseq(d)
            
            if exem:
                with open(fn_dir+'fn1.7.exemplar.train.syntaxnet.conll') as f:
                    d = f.readlines()
                exem_d = conll2tagseq(d)
            with open(fn_dir+'fn1.7.dev.syntaxnet.conll') as f:
                d = f.readlines()
            dev_d = conll2tagseq(d)
            with open(fn_dir+'fn1.7.test.syntaxnet.conll') as f:
                d = f.readlines()
            tst_d = conll2tagseq(d)
    else:
        if language == 'ko':
            if path == False:
                fn_dir = '/disk/data/corpus/koreanPropBank/original/'
            else:
                fn_dir = path
            if srl == 'propbank-dp':
                
                with open(fn_dir+'srl.dp_based.train.conll') as f:
                    d = f.readlines()
                trn_d = conll2tagseq(d)
                with open(fn_dir+'srl.dp_based.test.conll') as f:
                    d = f.readlines()
                tst_d = conll2tagseq(d)
                dev_d = []
            else:
                with open(fn_dir+'srl.span_based.train.conll') as f:
                    d = f.readlines()
                trn_d = conll2tagseq(d)
                with open(fn_dir+'srl.span_based.test.conll') as f:
                    d = f.readlines()
                tst_d = conll2tagseq(d)
                dev_d = []
    trn_d = data2tgt_data(trn_d, mode='train')
    if language == 'en':
        if exem:
            exem_d = data2tgt_data(exem_d, mode='train')
    tst = data2tgt_data(tst_d, mode='train')
    if dev_d:
        dev = data2tgt_data(dev_d, mode='train')
    else:
        dev = []
        
#     too_long_in_exem = [35285, 35286, 58002, 77448, 77993, 82010, 82061, 98118, 120524, 153131]
# #     too_long_in_exem = []
#     new_exem = []
#     for idx in range(len(exem)):
#         if idx in too_long_in_exem:
#             pass
#         else:
#             item = exem[idx]
#             new_exem.append(item)
        
#     trn += new_exem

    if language == 'en':
        if exem == True:
            ori_trn = trn_d + exem_d
            trn = []    

            too_long = [35285, 35286, 58002, 77448, 77993, 82010, 82061, 98118, 120524, 153131]
            for idx in range(len(ori_trn)):
                if idx in too_long:
                    pass
                else:
                    item = ori_trn[idx]
                    trn.append(item)
        else:
            trn = trn_d
    else:
        trn = trn_d
    
        
    print('# of instances in trn:', len(trn))
    print('# of instances in dev:', len(dev))
    print('# of instances in tst:', len(tst))
    print('data example:', trn[0])
    
    return trn, dev, tst
                
def data2tgt_data(input_data, mode=False):
    result = []
    for item in input_data:
        
        if mode == 'train':
            ori_tokens, ori_preds, ori_senses, ori_args = item[0],item[1],item[2],item[3]
        else:
            ori_tokens, ori_preds = item[0],item[1]
        for idx in range(len(ori_preds)):
            pred = ori_preds[idx]
            if pred != '_':
                if idx == 0:
                    begin = idx
                elif ori_preds[idx-1] == '_':
                    begin = idx
                end = idx
                
        tokens, preds, senses, args = [],[],[],[]
        for idx in range(len(ori_preds)):
            token = ori_tokens[idx]
            pred = ori_preds[idx]
            if mode == 'train':
                sense = ori_senses[idx]
                arg = ori_args[idx]
                
            if idx == begin:
                tokens.append('<tgt>')
                preds.append('_')
                if mode == 'train':
                    senses.append('_')
                    args.append('X')

            tokens.append(token)
            preds.append(pred)
            
            if mode == 'train':
                senses.append(sense)
                args.append(arg)

            if idx == end:
                tokens.append('</tgt>')
                preds.append('_')
                
                if mode == 'train':
                    senses.append('_')
                    args.append('X')
        sent = []
        sent.append(tokens)
        sent.append(preds)
        if mode == 'train':
            sent.append(senses)
            sent.append(args)
            
        result.append(sent)
    return result

def preprocessor(input_data):
    if type(input_data) == str:
        data = input_data.split(' ')
        result = []
        result.append(data)
    elif type(input_data) == list:
        result = input_data
    else:
        data = input_data['text'].split(' ')
        result = []
        result.append(data)
    return result
             
    
def get_frame_lu(frames, lus):
    frame = False
    for i in range(len(frames)):
        f = frames[i]
        if f != '_':
            frame = f
            lu = lus[i]
    return frame, lu
            
def remove_josa(phrase):
    tokens = phrase.split(' ')

    result = []
    for i in range(len(tokens)):
        token = tokens[i]
        if i < len(tokens)-1:
            result.append(token)
        else:
            m = kkma.pos(tokens[i])
            if m[-1][-1].startswith('J'):
                m.pop(-1)
                token = ''.join([t for t,p in m])
            result.append(token)
    result = ' '.join(result)
    return result
            
def frame2rdf(frame_conll, sent_id=False):
    triples = []
    for anno in frame_conll:
        tokens, lus, frames, args = anno[0],anno[1],anno[2],anno[3]
        frame, lu = get_frame_lu(frames, lus)
        if frame:
            if sent_id:
                triple = ('frame:'+frame+'#'+str(sent_id), 'frdf:lu', lu)
            else:
                triple = ('frame:'+frame, 'frdf:lu', lu)
            triples.append(triple)

        if frame:
            sbj = False
            pred_obj_tuples = []
            
            for idx in range(len(args)):
                arg_tag = args[idx]
                arg_tokens = []
                if arg_tag.startswith('B'):
                    fe_tag = arg_tag.split('-')[1]
                    arg_tokens.append(tokens[idx])
                    next_idx = idx + 1
                    while next_idx < len(args) and args[next_idx] == 'I-'+fe_tag:
                        arg_tokens.append(tokens[next_idx])
                        next_idx +=1
                    arg_text = ' '.join(arg_tokens)
                    arg_text = remove_josa(arg_text)
                    fe = fe_tag

                    # string to xsd
                    if fe == 'Time':
                        arg_text = kotimex.time2xsd(arg_text)
                    else:
                        pass
#                         arg_text = '\"'+arg_text+'\"'+'^^xsd:string'

                    rel = 'frdf:'+frame+'-'+fe

                    if rel == 'S':
                        pass
                    else:
                        p = rel
                        o = arg_text
                        pred_obj_tuples.append( (p,o) )

            for p, o in pred_obj_tuples:
                if sbj:
                    s = sbj
                else:
                    if sent_id:
                        s = 'frame:'+frame+'#'+str(sent_id)
                    else:
                        s = 'frame:'+frame
                triple = (s, p, o)
                triples.append(triple)
    return triples


        
    