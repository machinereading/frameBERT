from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')
from frameBERT.src import dataio
from frameBERT.src import utils
from torch.nn.parameter import Parameter
from transformers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

class FrameIdentifier(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, lufrmap=None, frargmap=None, masking=True, return_pooled_output=False, original_loss=False):
        super(FrameIdentifier, self).__init__(config)
        self.masking = masking
        self.num_senses = num_senses # total number of all frames
        self.num_args = num_args # total number of all frames
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.lufrmap = lufrmap # mapping table for lu to its frame candidates    
        self.frargmap = frargmap # mapping table for lu to its frame candidates
        self.return_pooled_output = return_pooled_output
        self.original_loss = original_loss   
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None, using_gold_fame=False, position_ids=None, head_mask=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        
        sense_logits = self.sense_classifier(pooled_output)      

        lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(device)
        
        sense_loss = 0 # loss for sense id
        arg_loss = 0 # loss for arg id
        
        if senses is not None:
            for i in range(len(sense_logits)):
                sense_logit = sense_logits[i]            

                lufr_mask = lufr_masks[i]
                    
                gold_sense = senses[i]
                gold_arg = args[i]
                
                #train sense classifier
                loss_fct_sense = CrossEntropyLoss(weight = lufr_mask)
                loss_per_seq_for_sense = loss_fct_sense(sense_logit.view(-1, self.num_senses), gold_sense.view(-1))
                sense_loss += loss_per_seq_for_sense

            total_loss = sense_loss
            loss = total_loss / len(sense_logits)
            
            if self.return_pooled_output:
                return pooled_output, loss
            else:
                return loss
        else:
            if self.return_pooled_output:
                return pooled_output, sense_logits, arg_logits
            else:
                return sense_logits, arg_logits
            
class ArgumentIdentifier(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, lufrmap=None, frargmap=None, masking=True, return_pooled_output=False, original_loss=False):
        super(ArgumentIdentifier, self).__init__(config)
        self.masking = masking
        self.num_senses = num_senses # total number of all frames
        self.num_args = num_args # total number of all frames
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.arg_classifier = nn.Linear(config.hidden_size, num_args)
        self.lufrmap = lufrmap # mapping table for lu to its frame candidates    
        self.frargmap = frargmap # mapping table for lu to its frame candidates
        self.return_pooled_output = return_pooled_output
        self.original_loss = original_loss   
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None, using_gold_fame=False, position_ids=None, head_mask=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        sense_logits = self.sense_classifier(pooled_output)
        arg_logits = self.arg_classifier(sequence_output)        

        lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(device)
        
        sense_loss = 0 # loss for sense id
        arg_loss = 0 # loss for arg id
        
        if senses is not None:
            for i in range(len(sense_logits)):
                sense_logit = sense_logits[i]
                arg_logit = arg_logits[i]                

                lufr_mask = lufr_masks[i]
                    
                gold_sense = senses[i]
                gold_arg = args[i]
                
                #train sense classifier
                loss_fct_sense = CrossEntropyLoss(weight = lufr_mask)
                loss_per_seq_for_sense = loss_fct_sense(sense_logit.view(-1, self.num_senses), gold_sense.view(-1))
                sense_loss += loss_per_seq_for_sense
                
                #train arg classifier
                masked_sense_logit = utils.masking_logit(sense_logit, lufr_mask)
                pred_sense, sense_score = utils.logit2label(masked_sense_logit)

                frarg_mask = utils.get_masks([gold_sense], self.frargmap, num_label=self.num_args, masking=True).to(device)[0]                
                loss_fct_arg = CrossEntropyLoss(weight = frarg_mask)

                
                # only keep active parts of loss
                if attention_mask is not None:
                    active_loss = attention_mask[i].view(-1) == 1
                    active_logits = arg_logit.view(-1, self.num_args)[active_loss]
                    active_labels = gold_arg.view(-1)[active_loss]
                    loss_per_seq_for_arg = loss_fct_arg(active_logits, active_labels)
                else:
                    loss_per_seq_for_arg = loss_fct_arg(arg_logit.view(-1, self.num_args), gold_arg.view(-1))
                arg_loss += loss_per_seq_for_arg

            total_loss = arg_loss
            loss = total_loss / len(sense_logits)
            
            if self.return_pooled_output:
                return pooled_output, loss
            else:
                return loss
        else:
            if self.return_pooled_output:
                return pooled_output, sense_logits, arg_logits
            else:
                return sense_logits, arg_logits
            


class ArgumentBoundaryIdentifier(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, lufrmap=None, frargmap=None, 
                 masking=True, return_pooled_output=False, original_loss=False, 
                 joint=True):
        super(ArgumentBoundaryIdentifier, self).__init__(config)
        self.masking = masking
        self.joint = joint
        self.num_senses = num_senses # total number of all frames
        self.num_args = num_args # total number of all frames
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.arg_classifier = nn.Linear(config.hidden_size, num_args)
        self.lufrmap = lufrmap # mapping table for lu to its frame candidates    
        self.frargmap = frargmap # mapping table for lu to its frame candidates
        self.return_pooled_output = return_pooled_output
        self.original_loss = original_loss   
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None, using_gold_fame=False, position_ids=None, head_mask=None):
        
        sense_loss = 0 # loss for sense id
        arg_loss = 0 # loss for arg id
        
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        
        if self.joint:
            pooled_output = self.dropout(pooled_output)
            sense_logits = self.sense_classifier(pooled_output)
            lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(device)
            masked_sense_logits = sense_logits * lufr_masks
            
        arg_logits = self.arg_classifier(sequence_output)
        
        # train frame identifier
        if self.joint:
            if senses is not None:
                loss_fct_sense = CrossEntropyLoss()
                loss_sense = loss_fct_sense(masked_sense_logits.view(-1, self.num_senses), senses.view(-1))
        
        # train arg classifier
        if senses is not None:
            loss_fct_arg = CrossEntropyLoss()        
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = arg_logits.view(-1, self.num_args)[active_loss]
                active_labels = args.view(-1)[active_loss]
                loss_arg = loss_fct_arg(active_logits, active_labels)
            else:
                loss_arg = loss_fct_arg(arg_logits.view(-1, self.num_args), args.view(-1))           
        
        
        if senses is not None:
            
            # joint vs only argument identification
            if self.joint:
                loss = 0.5*loss_sense + 0.5*loss_arg
            else:
                loss = loss_arg
                
            if self.return_pooled_output:
                return pooled_output, loss
            else:
                return loss
        else:
            if self.return_pooled_output:
                return pooled_output, masked_sense_logits, arg_logits
            else:
                return masked_sense_logits, arg_logits

class BertForJointShallowSemanticParsing(BertPreTrainedModel):
    def __init__(self, config, num_senses=2, num_args=2, lufrmap=None, frargmap=None, masking=True, return_pooled_output=False, original_loss=False):
        super(BertForJointShallowSemanticParsing, self).__init__(config)
        self.masking = masking
        self.num_senses = num_senses # total number of all frames
        self.num_args = num_args # total number of all frames
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sense_classifier = nn.Linear(config.hidden_size, num_senses)
        self.arg_classifier = nn.Linear(config.hidden_size, num_args)
        self.lufrmap = lufrmap # mapping table for lu to its frame candidates    
        self.frargmap = frargmap # mapping table for lu to its frame candidates
        self.return_pooled_output = return_pooled_output
        self.original_loss = original_loss   
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, lus=None, senses=None, args=None, using_gold_fame=False, position_ids=None, head_mask=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        sense_logits = self.sense_classifier(pooled_output)
        arg_logits = self.arg_classifier(sequence_output)        

        lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(device)
        
        sense_loss = 0 # loss for sense id
        arg_loss = 0 # loss for arg id
        
        if senses is not None:
            for i in range(len(sense_logits)):
                sense_logit = sense_logits[i]
                arg_logit = arg_logits[i]                

                lufr_mask = lufr_masks[i]
                    
                gold_sense = senses[i]
                gold_arg = args[i]
                
                #train sense classifier
                loss_fct_sense = CrossEntropyLoss(weight = lufr_mask)
                loss_per_seq_for_sense = loss_fct_sense(sense_logit.view(-1, self.num_senses), gold_sense.view(-1))
                sense_loss += loss_per_seq_for_sense
                
                #train arg classifier
                masked_sense_logit = utils.masking_logit(sense_logit, lufr_mask)
                pred_sense, sense_score = utils.logit2label(masked_sense_logit)

                frarg_mask = utils.get_masks([pred_sense], self.frargmap, num_label=self.num_args, masking=True).to(device)[0]                
                loss_fct_arg = CrossEntropyLoss(weight = frarg_mask)

                
                # only keep active parts of loss
                if attention_mask is not None:
                    active_loss = attention_mask[i].view(-1) == 1
                    active_logits = arg_logit.view(-1, self.num_args)[active_loss]
                    active_labels = gold_arg.view(-1)[active_loss]
                    loss_per_seq_for_arg = loss_fct_arg(active_logits, active_labels)
                else:
                    loss_per_seq_for_arg = loss_fct_arg(arg_logit.view(-1, self.num_args), gold_arg.view(-1))
                arg_loss += loss_per_seq_for_arg

            total_loss = 0.5*sense_loss + 0.5*arg_loss
            loss = total_loss / len(sense_logits)
            
            if self.return_pooled_output:
                return pooled_output, loss
            else:
                return loss
        else:
            if self.return_pooled_output:
                return pooled_output, sense_logits, arg_logits
            else:
                return sense_logits, arg_logits
            
            # masking sense logits
#             lufr_masks = utils.get_masks(lus, self.lufrmap, num_label=self.num_senses, masking=self.masking).to(device)        
#             masked_sense_logits = sense_logits * lufr_masks
            
#             # masking arg logits
#             pred_senses = []
#             for i in range(len(masked_sense_logits)):
#                 masked_sense_logit = masked_sense_logits[i]                    
#                 pred_sense, sense_score = utils.logit2label(masked_sense_logit)
#                 pred_senses.append(pred_sense)
#             frarg_mask = utils.get_masks(pred_senses, self.frargmap, num_label=self.num_args, masking=True).to(device)

#             frarg_mask = frarg_mask.view(len(frarg_mask), 1, -1)
#             frarg_mask = frarg_mask.repeat(1, len(arg_logits[0]), 1)

#             masked_arg_logits = arg_logits * frarg_mask        
            
#             if self.return_pooled_output:
#                 return pooled_output, masked_sense_logits, masked_arg_logits
#             else:
#                 return masked_sense_logits, masked_arg_logits