#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import re, os, pickle
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from math import log10


# In[85]:


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_sentence(filename):
    file = open(filename, 'r')
    data = file.read()
    sent = data.split("\n\n")
    print("No of Sentences: {}".format(len(sent)))
    return sent

def HMM_Train(sentences):
    Words = []
    nTagFreq = {}
    A = {}
    B = {}
    nTagFreq["<Start>"] = len(sentences)
    
    for i in range(len(sentences)):
        sent = sentences[i].split("\n")
        prev_tag = "<Start>"
        for j in range(len(sent)):
            word, tag = sent[j].split("\t")
            if tag in B:
                if word in B[tag]:
                    B[tag][word] += 1
                else:
                    B[tag][word] = 1
            else:
                B[tag] = {}
                B[tag][word] = 1
            
            if tag in nTagFreq:
                nTagFreq[tag] += 1
            else:
                nTagFreq[tag] = 1
            
            if prev_tag in A:
                if tag in A[prev_tag]:
                    A[prev_tag][tag] += 1
                else:
                    A[prev_tag][tag] = 1
            else:
                A[prev_tag] = {}
                A[prev_tag][tag] = 1
                
            prev_tag = tag
            if word not in Words:
                Words.append(word)
        if (prev_tag in A):
            if ("<End>" in A[prev_tag]):
                A[prev_tag]["<End>"] += 1
            else:
                A[prev_tag]["<End>"] = 1
        else:
            A[prev_tag] = {}
            A[prev_tag]["<End>"] = 1
    save_obj(A, "A")
    save_obj(B, "B")
    save_obj(nTagFreq, "nTagFreq")
    save_obj(Words, "Words")
    
def Viterbi(sentence):
    Words = load_obj("Words")
    nTagFreq = load_obj("nTagFreq")
    A = load_obj("A")
    B = load_obj("B")
    VocabSize = len(Words)
    StartCount = nTagFreq["<Start>"]
    del(nTagFreq["<Start>"])
    Tags = list(nTagFreq.keys())
    TagSize = len(list(Tags))
    
    sent = sentence.split("\n")
    Vt = np.zeros((TagSize, len(sent)))
    VtStar = np.zeros((TagSize, len(sent)), dtype=np.int64)
    POSTag = ['NULL' for i in range(len(sent))]
    
    # Start State Handling
    prev_tag = "<Start>"
    for i in range(TagSize):
        w = sent[0]
        p=0
        if (Tags[i] in A[prev_tag]):
            p += log10((A[prev_tag][Tags[i]] + 1)/(StartCount + TagSize))
        else:
            p += log10(1/(StartCount + TagSize))
        if (w in B[Tags[i]]):
            p += log10((B[Tags[i]][w] + 1)/(nTagFreq[Tags[i]] + VocabSize))
        else:
            p += log10(1/(nTagFreq[Tags[i]] + VocabSize))
        Vt[i, 0] = p
        VtStar[i, 0] = -1
    
    # Mid States
    for i in range(1, len(sent)):
        for j in range(TagSize):
            probs = np.zeros(TagSize)
            for k in range(TagSize):
                p = 0
                w = sent[i]
                if (Tags[k] in A):
                    if (Tags[j] in A[Tags[k]]):
                        p += log10((A[Tags[k]][Tags[j]] + 1)/(nTagFreq[Tags[k]] + TagSize))
                    else:
                        p += log10(1/(nTagFreq[Tags[k]] + TagSize))
                else:
                    p += log10(1/TagSize)
                if (w in B[Tags[j]]):
                    p += log10((B[Tags[j]][w] + 1)/(nTagFreq[Tags[j]] + VocabSize))
                else:
                    p += log10(1/(nTagFreq[Tags[j]] + VocabSize))
                probs[k] = Vt[k, i-1] + p
            Vt[j, i] = np.amax(probs)
            VtStar[j, i] = np.argmax(probs)
    
    # EndState 
    prob = np.zeros(TagSize)
    for i in range(TagSize):
        prob[i] = Vt[i, len(sent)-1]
        if (Tags[i] in A):
            if ("<End>" in A[Tags[i]]):
                prob[i] += log10((A[Tags[i]]["<End>"] + 1)/(nTagFreq[Tags[i]] + TagSize))
            else:
                prob[i] += log10(1/(nTagFreq[Tags[i]] + TagSize))
        else:
            prob[i] += log10(1/TagSize)
    
    POSTag[len(sent)-1] = Tags[np.argmax(prob)]
    index = np.argmax(Vt[:, len(sent)-1])
    for i in range(len(sent)-2, -1, -1):
        POSTag[i] = Tags[index]
        index = VtStar[index, i]
    return POSTag


def ViterbiValidation(sentence):
    Words = load_obj("Words")
    nTagFreq = load_obj("nTagFreq")
    A = load_obj("A")
    B = load_obj("B")
    VocabSize = len(Words)
    StartCount = nTagFreq["<Start>"]
    del(nTagFreq["<Start>"])
    Tags = list(nTagFreq.keys())
    TagSize = len(list(Tags))
    
    sent = sentence.split("\n")
    Vt = np.zeros((TagSize, len(sent)))
    VtStar = np.zeros((TagSize, len(sent)), dtype=np.int64)
    POSTag = ['NULL' for i in range(len(sent))]
    
    # Start State Handling
    prev_tag = "<Start>"
    for i in range(TagSize):
        w = sent[0].split("\t")[0]
        p=0
        if (Tags[i] in A[prev_tag]):
            p += log10((A[prev_tag][Tags[i]] + 1)/(StartCount + TagSize))
        else:
            p += log10(1/(StartCount + TagSize))
        if (w in B[Tags[i]]):
            p += log10((B[Tags[i]][w] + 1)/(nTagFreq[Tags[i]] + VocabSize))
        else:
            p += log10(1/(nTagFreq[Tags[i]] + VocabSize))
        Vt[i, 0] = p
        VtStar[i, 0] = -1
    
    # Mid States
    for i in range(1, len(sent)):
        for j in range(TagSize):
            probs = np.zeros(TagSize)
            for k in range(TagSize):  # prev_tag
                p = 0
                w = sent[i].split("\t")[0]
                if (Tags[k] in A):
                    if (Tags[j] in A[Tags[k]]):
                        p += log10((A[Tags[k]][Tags[j]] + 1)/(nTagFreq[Tags[k]] + TagSize))
                    else:
                        p += log10(1/(nTagFreq[Tags[k]] + TagSize))
                else:
                    p += log10(1/TagSize)
                if (w in B[Tags[j]]):
                    p += log10((B[Tags[j]][w] + 1)/(nTagFreq[Tags[j]] + VocabSize))
                else:
                    p += log10(1/(nTagFreq[Tags[j]] + VocabSize))
                probs[k] = Vt[k, i-1] + p
            Vt[j, i] = np.amax(probs)
            VtStar[j, i] = np.argmax(probs)
    
    # EndState 
    prob = np.zeros(TagSize)
    for i in range(TagSize):
        prob[i] = Vt[i, len(sent)-1]
        if (Tags[i] in A):
            if ("<End>" in A[Tags[i]]):
                prob[i] += log10((A[Tags[i]]["<End>"] + 1)/(nTagFreq[Tags[i]] + TagSize))
            else:
                prob[i] += log10(1/(nTagFreq[Tags[i]] + TagSize))
        else:
            prob[i] += log10(1/TagSize)
    
    POSTag[len(sent)-1] = Tags[np.argmax(prob)]
    index = VtStar[np.argmax(prob), len(sent)-1]
    for i in range(len(sent)-2, -1, -1):
        POSTag[i] = Tags[index]
        index = VtStar[index, i]
    return POSTag
    


# In[86]:


Data = "Data/"
sents = get_sentence(Data + "Training set_HMM.txt")
train = sents[:int(len(sents)*0.99)]
test = sents[int(len(sents)*.99):]


# In[87]:


HMM_Train(sents)


# In[97]:


sents = get_sentence("Test.in")
file = open("Test.out", 'w')
for i in range(len(sents)):
    Tags = Viterbi(sents[i])
    lines = sents[i].split("\n")
    for j in range(len(lines)):
        file.write(lines[j] + "\t" + Tags[j] + "\n")
    file.write("\n")
file.close()


# In[90]:


correct = 0
total = 0
for i in tqdm(range(len(test))):
    Tags = ViterbiValidation(test[i])
    lines = test[i].split("\n")
    for j in range(len(lines)):
        if (lines[j].split("\t")[1] == Tags[j]):
#             print(lines[j].split("\t")[1], Tags[j], "+")
            correct += 1
#         else:
#             print(lines[j].split("\t")[1], Tags[j], "-")
        total += 1
print("Accuracy: {}".format((correct/total)*100))


# In[ ]:




