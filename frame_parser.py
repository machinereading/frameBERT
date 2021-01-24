
# coding: utf-8

# In[1]:


import sys
import glob
import torch
sys.path.append('../')
import os
import numpy as np
from transformers import *
from frameBERT.src import utils as frameBERT_utils
from frameBERT.src import dataio
from frameBERT import target_identifier
from frameBERT import inference
from frameBERT.src.modeling import *
from frameBERT.koreanframenet.src import conll2textae
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm, trange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
# if device != "cpu":
#     torch.cuda.set_device(device)

print('\n###DEVICE:', device)


# In[30]:


class FrameParser():
    def __init__(self, fnversion=1.2, language='ko',masking=True, srl='framenet', 
                 model_path=False, gold_pred=False, viterbi=False, tgt=True, 
                 pretrained='bert-base-multilingual-cased', info=True, only_lu=True, adj=True):
        self.fnversion = fnversion
        self.language = language
        self.masking = masking
        self.srl = srl
        self.gold_pred = gold_pred
        self.viterbi = viterbi
        self.pretrained = pretrained
        self.tgt = tgt #using <tgt> and </tgt> as a special token
        self.only_lu = only_lu
        self.adj = adj
        
        if self.masking==True:
            self.targetid = target_identifier.targetIdentifier(language=self.language, only_lu=self.only_lu, adj=self.adj)
        else:
            self.targetid = target_identifier.targetIdentifier(language=self.language, only_lu=False, masking=self.masking, adj=self.adj)
            
        if self.srl == 'propbank-dp':
            self.viterbi = False
            self.masking = False
        
        if info:
            print('srl model:', self.srl)
            print('language:', self.language)
            print('version:', self.fnversion)
            print('using viterbi:', self.viterbi)
            print('using masking:', self.masking)
            print('pretrained BERT:', self.pretrained)
            print('using TGT special token:', self.tgt)
        
        self.bert_io = frameBERT_utils.for_BERT(mode='predict', srl=self.srl, language=self.language, 
                              masking=self.masking, fnversion=self.fnversion,
                              pretrained=self.pretrained, info=info)  
        
        #load model
        if model_path:
            self.model_path = model_path
        else:
            print('model_path={your_model_dir}')
#         self.model = torch.load(model_path, map_location=device)

        if self.srl == 'framenet':
            self.model = BertForJointShallowSemanticParsing.from_pretrained(self.model_path, 
                                                                       num_senses = len(self.bert_io.sense2idx), 
                                                                       num_args = len(self.bert_io.bio_arg2idx),
                                                                       lufrmap=self.bert_io.lufrmap, masking=self.masking,
                                                                       frargmap = self.bert_io.bio_frargmap)
        elif self.srl == 'framenet-argid':
            self.model = ArgumentBoundaryIdentifier.from_pretrained(self.model_path, 
                                                                    num_senses = len(self.bert_io.sense2idx), 
                                                                    num_args = len(self.bert_io.bio_argument2idx),
                                                                    lufrmap=self.bert_io.lufrmap, 
                                                                    frargmap = self.bert_io.bio_frargmap)
        else:
            self.model = BertForJointShallowSemanticParsing.from_pretrained(self.model_path, 
                                                                       num_senses = len(self.bert_io.sense2idx), 
                                                                       num_args = len(self.bert_io.bio_arg2idx),
                                                                       lufrmap=self.bert_io.lufrmap, masking=self.masking,
                                                                       frargmap = self.bert_io.bio_frargmap)
        
            
            
            
        self.model.to(device)
        if info:
            print('...loaded model path:', self.model_path)
#         self.model = BertForJointShallowSemanticParsing
        self.model.eval()
        if info:
            print(self.model_path)
            print('...model is loaded')
        
        # trainsition parameter for vitervi decoding
        if self.srl != 'propbank-dp':
            self.transition_param = inference.get_transition_params(self.bert_io.idx2bio_arg.values())
        
    def parser(self, input_d, sent_id=False, result_format=False, frame_candis=5):
        input_conll = dataio.preprocessor(input_d)
        
        #target identification
        if self.gold_pred:
            if len(input_conll[0]) == 2:
                pass
            else:
                input_conll = [input_conll]
            tgt_data = input_conll
        else:
            if self.srl == 'framenet':
                tgt_conll = self.targetid.target_id(input_conll)
            else:
                tgt_conll = self.targetid.pred_id(input_conll)
        
            # add <tgt> and </tgt> to target word
            tgt_data = dataio.data2tgt_data(tgt_conll, mode='parse')

        if tgt_data:
            
            # convert conll to bert inputs
            bert_inputs = self.bert_io.convert_to_bert_input_JointShallowSemanticParsing(tgt_data)
            dataloader = DataLoader(bert_inputs, sampler=None, batch_size=1)
            
            pred_senses, pred_args = [],[]
            sense_candis_list = []
            for batch in dataloader:
#                 torch.cuda.set_device(device)
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_orig_tok_to_maps, b_lus, b_token_type_ids, b_masks = batch
                
                with torch.no_grad():
                    tmp_eval_loss = self.model(b_input_ids, lus=b_lus, 
                                               token_type_ids=b_token_type_ids, attention_mask=b_masks)
                    sense_logits, arg_logits = self.model(b_input_ids, lus=b_lus, 
                                                          token_type_ids=b_token_type_ids, attention_mask=b_masks)
                

                lufr_masks = frameBERT_utils.get_masks(b_lus, 
                                             self.bert_io.lufrmap, 
                                             num_label=len(self.bert_io.sense2idx), 
                                             masking=self.masking).to(device)

                
                b_input_ids_np = b_input_ids.detach().cpu().numpy()
                arg_logits_np = arg_logits.detach().cpu().numpy()
                    
                b_input_ids, arg_logits = [],[]
                
                for b_idx in range(len(b_orig_tok_to_maps)):
                    orig_tok_to_map = b_orig_tok_to_maps[b_idx]
                    bert_token = self.bert_io.tokenizer.convert_ids_to_tokens(b_input_ids_np[b_idx])
                    tgt_idx = frameBERT_utils.get_tgt_idx(bert_token, tgt=self.tgt)
                    
                    input_id, sense_logit, arg_logit = [],[],[]

                    for idx in orig_tok_to_map:                        
                        if idx != -1:
                            if idx not in tgt_idx:
                                try:
                                    input_id.append(b_input_ids_np[b_idx][idx])
                                    arg_logits_np[b_idx][idx][1] = np.NINF
                                    arg_logit.append(arg_logits_np[b_idx][idx])
                                except KeyboardInterrupt:
                                    raise
                                except:
                                    pass
                                
                    b_input_ids.append(input_id)
                    arg_logits.append(arg_logit)
                    
                b_input_ids = torch.Tensor(b_input_ids).to(device)
                arg_logits = torch.Tensor(arg_logits).to(device)
                    
                for b_idx in range(len(sense_logits)):
                    input_id = b_input_ids[b_idx]
                    sense_logit = sense_logits[b_idx]
                    arg_logit = arg_logits[b_idx]
                    
                    lufr_mask = lufr_masks[b_idx]                        
                    masked_sense_logit = frameBERT_utils.masking_logit(sense_logit, lufr_mask)
                    pred_sense, sense_score = frameBERT_utils.logit2label(masked_sense_logit)
                    
                    sense_candis = frameBERT_utils.logit2candis(masked_sense_logit, 
                                                      candis=frame_candis, 
                                                      idx2label=self.bert_io.idx2sense)
                    sense_candis_list.append(sense_candis)
                    
                    if self.srl == 'framenet':
                        arg_logit_np = arg_logit.detach().cpu().numpy()
                        arg_logit = []
                        frarg_mask = frameBERT_utils.get_masks([pred_sense], 
                                                     self.bert_io.bio_frargmap, 
                                                     num_label=len(self.bert_io.bio_arg2idx), 
                                                     masking=True).to(device)[0]
                        for logit in arg_logit_np:
                            masked_logit = frameBERT_utils.masking_logit(logit, frarg_mask)
                            arg_logit.append(np.array(masked_logit))
                        arg_logit = torch.Tensor(arg_logit).to(device)
                    else:
                        pass
                    
                    pred_arg = []
                    for logit in arg_logit:
                        label, score = frameBERT_utils.logit2label(logit)
                        pred_arg.append(int(label))
                        
                    pred_senses.append([int(pred_sense)])
                    pred_args.append(pred_arg)

            pred_sense_tags = [self.bert_io.idx2sense[p_i] for p in pred_senses for p_i in p]
            if self.srl == 'framenet':
                pred_arg_tags = [[self.bert_io.idx2bio_arg[p_i] for p_i in p] for p in pred_args]
            elif self.srl == 'framenet-argid':
                pred_arg_tags = [[self.bert_io.idx2bio_argument[p_i] for p_i in p] for p in pred_args]
            else:
                pred_arg_tags = [[self.bert_io.idx2bio_arg[p_i] for p_i in p] for p in pred_args]

            conll_result = []

            for i in range(len(pred_arg_tags)):       
                
                raw = tgt_data[i]
                
                conll, toks, lus = [],[],[]
                for idx in range(len(raw[0])):
                    tok, lu = raw[0][idx], raw[1][idx]
                    if tok == '<tgt>' or tok == '</tgt>':
                        pass
                    else:
                        toks.append(tok)
                        lus.append(lu)
                conll.append(toks)
                conll.append(lus)
                
                sense_seq = ['_' for i in range(len(conll[1]))]
                for idx in range(len(conll[1])):
                    if conll[1][idx] != '_':
                        sense_seq[idx] = pred_sense_tags[i]
                        
                conll.append(sense_seq)
                conll.append(pred_arg_tags[i])
                
                conll_result.append(conll)
        else:
            conll_result = []
        
        result = []
        if result_format == 'all':            
            result = {}
            result['conll'] = conll_result

            if conll_result:
                textae = conll2textae.get_textae(conll_result)
                frdf = dataio.frame2rdf(conll_result, sent_id=sent_id)
                topk = dataio.topk(conll_result, sense_candis_list)
            else:
                textae = []
                frdf = []
                topk = {}
            result['textae'] = textae
            result['graph'] = frdf
            result['topk'] = topk
        elif result_format == 'textae':
            if conll_result:
                textae = conll2textae.get_textae(conll_result)
            else:
                textae = []
            result = textae
        elif result_format == 'graph':
            if conll_result:
                frdf = dataio.frame2rdf(conll_result, sent_id=sent_id, language=self.language)
            else:
                frdf = []
            result = frdf
        elif result_format == 'topk':
            if conll_result:
                topk = dataio.topk(conll_result, sense_candis_list)
            else:
                topk = {}
            result = topk
        else:
            result = conll_result
        
        return result     

