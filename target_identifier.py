
# coding: utf-8

# In[1]:


import os
import json
from collections import Counter
import jpype
import sys
sys.path.append('../')

from frameBERT.koreanframenet import koreanframenet
kfn = koreanframenet.interface(1.1)
# from konlpy.tag import Kkma
# kkma = Kkma()

try:
    target_dir = os.path.dirname( os.path.abspath( __file__ ))
except:
    target_dir = '.'

class targetIdentifier():
    def __init__(self, srl='framenet', language='ko', only_lu=True):
        self.srl = srl
        self.language = language
        self.only_lu = only_lu
        
        if self.language == 'ko':
            from konlpy.tag import Kkma
            self.kkma = Kkma()
            
            with open(target_dir+'/data/targetdic-1.1.json','r') as f:
                targetdic = json.load(f)
            self.targetdic = targetdic
        else:
            import nltk
            self.lemmatizer = nltk.WordNetLemmatizer()
            self.pos_tagger = nltk.pos_tag
            
            with open(target_dir+'/data/targetdic-FN1.7.json','r') as f:
                targetdic = json.load(f)
            self.targetdic = targetdic
    
    def targetize(self, word):
        jpype.attachThreadToJVM()
        target_candis = []
        morps = self.kkma.pos(word)
        v = False
        for m,p in morps:
            if p == 'XSV' or p == 'VV' or p == 'VA':
                v = True    
        if v:
            for i in range(len(morps)):
                m,p = morps[i]
                if p == 'VA' or p == 'VV':
                    if p == 'VV':
                        pos = 'v'
                    elif p == 'VA':
                        pos = 'a'
                    else:
                        pos = 'v'

                    if m[0] == word[0] and len(m) >= 1:
                        target_candis.append((m,pos))
                if p == 'NNG':
                    pos = 'n'
                    if m[0] == word[0] and len(m) >= 1:
                        target_candis.append((m,pos))
                if i > 0 and p == 'XSV':
                    pos = 'v'
                    if m[0] == word[0] and len(m) >= 1:
                        target_candis.append((m,pos))
                    r = morps[i-1][0]+m
                    if r[0] == word[0]:
                        target_candis.append((r,pos))
        else:
            pos = 'n'
            pos_list = []
            for m,p in morps:
                if p.startswith('J'):
                    pos_list.append(m)
                elif p == 'VCP' or p == 'EFN':
                    pos_list.append(m)
            for m, p in morps:
                if p == 'NNG':
                    if len(pos_list) == 0:
                        if m == word:
                            target_candis.append((m, pos))
                    else:
                        if m[0] == word[0]:
                            target_candis.append((m, pos))
                                
        return target_candis

    def get_lu_by_token(self, token):
        target_candis = self.targetize(token)
        lu_candis = []
        for target_candi, word_pos in target_candis:
            for lu in self.targetdic:
                if target_candi in self.targetdic[lu]:
                    lu_pos = lu.split('.')[-1]
                    if word_pos == lu_pos:
                        lu_candis.append(lu)
            if self.only_lu==False:
                lu_candis.append(target_candi+'.'+word_pos)
        common = Counter(lu_candis).most_common()
        if len(common) > 0:
            result = common[0][0]
        else:
            result = False
        return result
    
    def get_enlu(self, token, pos):
        result = False
        
        p = False
        if pos == 'NN' or pos == 'NNS':
            p = 'n'
        elif pos.startswith('V'):
            p = 'v'
        elif pos.startswith('J'):
            p = 'a'
        else:
            p = False
            
        # lemmatize       
            
        if p:
            lemma = self.lemmatizer.lemmatize(token, p)
            if lemma:
                for lu in self.targetdic:
                    lu_pos = lu.split('.')[-1]                    
                    if p == lu_pos:
                        candi = self.targetdic[lu]
                        if lemma in candi:
                            result = lu
                        else:
                            pass
                    
        return result
    
    def target_id(self, input_conll):
        if self.language == 'ko':
            result = []
            tokens = input_conll[0]
            for idx in range(len(tokens)):
                token = tokens[idx]
                lu = self.get_lu_by_token(token)
                lus = ['_' for i in range(len(tokens))]
                if lu:
                    lus[idx] = lu
                    instance = []            
                    instance.append(tokens)
                    instance.append(lus)
                    result.append(instance)
                    
        elif self.language == 'en':
            result = []
            tokens = input_conll[0]
            pos_tagged = self.pos_tagger(tokens)
            
            for idx in range(len(tokens)):
                token = tokens[idx]
                pos = pos_tagged[idx][-1]
                
                lu = self.get_enlu(token, pos)
                lus = ['_' for i in range(len(tokens))]
                if lu:
                    lus[idx] = lu
                    instance = []            
                    instance.append(tokens)
                    instance.append(lus)
                    result.append(instance)               
                
        else:
            result = []
                      
        return result
    
    def pred_id(self, input_conll):
        result = []
        tokens = input_conll[0]
        for idx in range(len(tokens)):
            token = tokens[idx]
            lus = ['_' for i in range(len(tokens))]
            target_candis = self.targetize(token)
            for target_candi, word_pos in target_candis:
                if word_pos == 'v' or word_pos == 'a':
                    lus[idx] = 'PRED'
                    instance = []
                    instance.append(tokens)
                    instance.append(lus)
                    result.append(instance)
        return result
