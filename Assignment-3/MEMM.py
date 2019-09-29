#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import re, os, pickle, operator
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from math import log10, exp


# In[88]:


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_sentence(filename):
    file = open(filename, 'r', encoding="utf8")
    data = file.read()
    sent = data.split("\n\n")
    print("No of Sentences: {}".format(len(sent)))
    return sent

def POSTagFreq(sents):
    POS = {}
    for i in range(len(sents)):
        sent = sents[i].split("\n")
        for j in range(len(sent)):
            word, tag, bio = sent[j].split("\t")
            if tag in POS:
                POS[tag] += 1
            else:
                POS[tag] = 1
    POS = sorted(POS.items(), key=lambda x: x[1], reverse=True)
    return POS[0]

def Train(sentences):
    MostOccTag = POSTagFreq(sents)
    WordCount = {}
    BIOTagFreq = {}
    F1 = {}
    F2 = {}
    F3 = {}
    F4 = {}
    F5 = {}
    for i in range(len(sentences)):
        sent = sentences[i].split("\n")
        for j in range(len(sent)):
            word, tag, bio = sent[j].split("\t")
            
            #Feature 1
            if j == 0:
                if bio in F1:
                    if word in F1[bio]:
                        F1[bio][word] += 1
                    else:
                        F1[bio][word] = 1
                else:
                    F1[bio] = {}
                    F1[bio][word] = 1
            #Feature 2
            if j == (len(sent)-1):
                if bio in F2:
                    if word in F2[bio]:
                        F2[bio][word] += 1
                    else:
                        F2[bio][word] = 1
                else:
                    F2[bio] = {}
                    F2[bio][word] = 1
                    
            # Feature 3
            if j > 0:
                _, prev_tag, _ = sent[j-1].split("\t")
                if bio in F3:
                    if word in F3[bio]:
                        if prev_tag in F3[bio][word]:
                            F3[bio][word][prev_tag] += 1
                        else:
                            F3[bio][word][prev_tag] = 1
                    else:
                        F3[bio][word] = {}
                        F3[bio][word][prev_tag] = 1
                else:
                    F3[bio] = {}
                    F3[bio][word] = {}
                    F3[bio][word][prev_tag] = 1
            
            # Feature 4
            if len(sent)-1 > j:
                _, next_tag, _ = sent[j+1].split("\t")
                if next_tag == MostOccTag:
                    if bio in F4:
                        if word in F4[bio]:
                            if next_tag in F4[bio][word]:
                                F4[bio][word][next_tag] += 1
                            else:
                                F4[bio][word][next_tag] = 1
                        else:
                            F4[bio][word] = {}
                            F4[bio][word][next_tag] = 1
                    else:
                        F4[bio] = {}
                        F4[bio][word] = {}
                        F4[bio][word][next_tag] = 1
                    
            # Feature 5
            if bio in F5:
                if word in F5[bio]:
                    F5[bio][word] += 1
                else:
                    F5[bio][word] = 1
            else:
                F5[bio] = {}
                F5[bio][word] = 1
            
            # Update Denominator
            if bio in  WordCount:
                if word in WordCount[bio]:
                    WordCount[bio][word] += 1
                else:
                    WordCount[bio][word] = 1
            else:
                WordCount[bio] = {}
                WordCount[bio][word] = 1
            
            if bio in BIOTagFreq:
                BIOTagFreq[bio] += 1
            else:
                BIOTagFreq[bio] = 1

    save_obj(MostOccTag, "MostOccTag")
    save_obj(WordCount, "WordCount")
    save_obj(BIOTagFreq, "BIOTagFreq")
    save_obj(F1, "F1")
    save_obj(F2, "F2")
    save_obj(F3, "F3")
    save_obj(F4, "F4")
    save_obj(F5, "F5")
    
def predict(sentences):
    MostOccTag = load_obj("MostOccTag")
    WordCount = load_obj("WordCount")
    BIOTagFreq = load_obj("BIOTagFreq")
    F1 = load_obj("F1")
    F2 = load_obj("F2")
    F3 = load_obj("F3")
    F4 = load_obj("F4")
    F5 = load_obj("F5")
    
    bioTag = []
    for i in range(len(sentences)):
        sent = sentences[i].split("\n")
        PredTag = []
        for j in range(len(sent)):
            word, tag, _ = sent[j].split("\t")
            proba = np.zeros(3)
            Tags = ["I-NP", "B-NP", "O"]
            for idx, bio in enumerate(Tags):
                wf = 0
                if j == 0:
                    if bio in F1:
                        if word in F1[bio]:
                            wf += (F1[bio][word]/WordCount[bio][word])
                if j == len(sent)-1:
                    if bio in F2:
                        if word in F2[bio]:
                            wf += (F2[bio][word]/WordCount[bio][word])
                if j > 0:
                    _, prev_tag, _ = sent[j-1].split("\t")
                    if bio in F3:
                        if word in F3[bio]:
                            if prev_tag in F3[bio][word]:
                                wf += (F3[bio][word][prev_tag]/WordCount[bio][word])
                if j < len(sent)-1:
                    if bio in F4:
                        if word in F4[bio]:
                            if MostOccTag in F4[bio][word]:
                                wf += (F4[bio][word][MostOccTag]/WordCount[bio][word])
                if bio in F5:
                    if word in F5[bio]:
                        wf += (F5[bio][word]/BIOTagFreq[bio])
                proba[idx] = exp(wf)
            PredTag.append(Tags[np.argmax(proba)])
        bioTag.append(PredTag)
    return bioTag


def save_output(sents, bioTag):
    file = open("output.np", "w")
    Tags = ["I-NP", "B-NP", "O"]
    CorrectCount = {}
    Counts = {}
    for i in range(len(Tags)):
        Counts[Tags[i]] = 0
        CorrectCount[Tags[i]] = 0
    for i in range(len(sents)):
        sent = sents[i].split("\n")
        for j in range(len(sent)):
            word, tag, bio = sent[j].split("\t")
            file.write(word + "\t" + tag + "\t" + bio + "\t" + bioTag[i][j] + "\n")
            Counts[bio] += 1
            if (bio == bioTag[i][j]):
                CorrectCount[bio] += 1
        file.write("\n")
    for i in range(len(Tags)):
        print("{} : {}".format(Tags[i], (CorrectCount[Tags[i]]/Counts[Tags[i]])*100 ))
            


# In[71]:


Data = "Data/"
sents = get_sentence(Data + "train.np")
test_sents = get_sentence(Data + "dev.np")


# In[72]:


# print(sents[0])
Train(sents)


# In[89]:


bioTag = predict(sents)
save_output(sents, bioTag)


# In[90]:


bioTag = predict(test_sents)
save_output(test_sents, bioTag)


# In[ ]:




