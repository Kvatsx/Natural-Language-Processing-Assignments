#!/usr/bin/env python
# coding: utf-8

# ### Kaustav Vats (2016048)

# In[41]:


import nltk
import pandas as pd
import numpy as np
import string, re
import json
from math import log
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm_notebook as tqdm
# References
# Cosine Similarity and TFIDF Matrix
# https://github.com/parulnith/Building-a-Simple-Chatbot-in-Python-using-NLTK/blob/master/Chatbot.ipynb
# https://stackoverflow.com/questions/15899861/efficient-term-document-matrix-with-nltk
# https://github.com/williamscott701/Information-Retrieval/blob/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score/TF-IDF.ipynb
# Doc2Vec
# https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
# Word2Vec and Doc2Vec
# https://shuzhanfan.github.io/2018/08/understanding-word2vec-and-doc2vec/
# https://ireneli.eu/2016/07/27/nlp-05-from-word2vec-to-doc2vec-a-simple-example-with-gensim/


# In[92]:


def load_data(filename):
    data = []
    f = open("Data/" + filename, 'r', encoding="utf8")
    for line in f:
        line = line.strip()
        line = SentPreProcessing(line)
        data.append(line)
    return data

def stemming(sent):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(sent))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def SentPreProcessing(sent):
    word_tokens = word_tokenize(sent)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    sent = " ".join(filtered_sentence)
    sent = stemming(sent)
    sent = sent.lower()
    for i in string.punctuation:
        sent = sent.replace(i, ' ')
    sent = re.sub(r'[^\w]', ' ', sent)
    sent = re.sub(r'\d+', '', sent)
    return sent

def getVocab(DocTokens, data):
    Vocab = set()
    for d in data:
        tkns = word_tokenize(d)
        DocTokens.append(tkns)
        for t in tkns:
            Vocab.add(t)
    Vocab = list(Vocab)
    return Vocab

def Get_tfidf_Matrix(data):
    DocTokens = []
    for d in data:
        tkns = word_tokenize(d)
        DocTokens.append(tkns)
        
    Vocab = getVocab(DocTokens, data)
        
    tfidf = np.zeros((len(data), len(Vocab)))
    for i in range(len(data)):
        for j in range(len(Vocab)):
            tfidf[i, j] = 1 + log(1 + DocTokens[i].count(Vocab[j]))
            
    N = len(data)
    IDF_Vector = []
    for i in range(len(Vocab)):
        w = Vocab[i]
        count = 0
        for j in range(N):
            if (w in DocTokens[j]):
                count += 1
        IDF_Vector.append(log(N/(count+1)))
        tfidf[:, i] = tfidf[:, i] * log(N/(count+1))
        
    return tfidf, Vocab, IDF_Vector

def get_tfidf_query(data, vocab, idf_v):
    DocTokens = []
    for d in data:
        tkns = word_tokenize(d)
        DocTokens.append(tkns)
    
    tfidf = np.zeros((len(data), len(vocab)))
    for i in range(len(data)):
        for j in range(len(vocab)):
            tfidf[i, j] = 1 + log(1 + DocTokens[i].count(vocab[j]))
    
    for i in range(len(vocab)):
        tfidf[:, i] = tfidf[:, i] * idf_v[i]
        
    return tfidf

def load_questions(filename):
    data = []
    f = open("Data/" + filename, 'r')
    for line in f:
        line = line.strip()
        data.append(json.loads(line))
    return data

def ShowAnalysis(Analysis, count):
    for i in range(count):
        print(Analysis[i])

def FindMax(S):
    maxi = S[0][1]
    for se in S:
        if (maxi < se[1]):
            maxi = se[1]
    return maxi
def Counts(Analysis):
    ABCD = [0, 0, 0, 0]
    for i in range(len(Analysis)):
        for sym in Analysis[i][2]:
            if (Analysis[i][1] != sym):
                if sym == 'A':
                    ABCD[0] += 1
                elif sym == 'B':
                    ABCD[1] += 1
                elif sym == 'C':
                    ABCD[2] += 1
                else:
                    ABCD[3] += 1
    return ABCD


# In[20]:


Data = load_data("data.txt")
Questions = load_questions("test.jsonl")


# ### Step 1  |  Cosine Similarity

# In[21]:


TermDocMat, Vocab, IDF_V = Get_tfidf_Matrix(Data)
print(TermDocMat.shape)


# In[22]:


Alpha = ["A", "B", "C", "D"]
QScores = []
for ques in tqdm(Questions):
    Q = ques["question"]["stem"]
    A = ques["question"]["choices"]
    C = ques["answerKey"]
    Scores = []
    for option in A:
        tempQ = Q + " " + option["text"]
        tempQ = [SentPreProcessing(tempQ)]
        temp_tfidf = get_tfidf_query(tempQ, Vocab, IDF_V)
        Scores.append(np.amax(cosine_similarity(temp_tfidf, TermDocMat)))
    setOfOptions = []
    maxi = max(Scores)
    for i in range(len(Scores)):
        if maxi == Scores[i]:
            setOfOptions.append(Alpha[i])
    if (C in setOfOptions):
        QScores.append(1/len(setOfOptions))
    else:
        QScores.append(0)
print("Accuracy:", (sum(QScores)/len(Questions))*100)


# ### Step 2     |     Doc2Vec

# In[43]:


TaggedData = []
for i, _d in enumerate(Data):
    tag_data = TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
    TaggedData.append(tag_data)


# In[76]:


max_epochs = 50
vec_size = 20
alpha = 0.025

'''
dm: If dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW). 
Distributed Memory model preserves the word order in a document whereas Distributed Bag of words just uses the bag of words approach, 
which doesn’t preserve any word order.
'''

model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=0)
model.build_vocab(TaggedData)


# In[77]:


for epoch in range(max_epochs):
    if epoch%10 == 0:
        print('iteration {}'.format(epoch))
    model.train(TaggedData, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

model.save("Data/Doc2Vec.model")
print("Done!")


# In[78]:


model = Doc2Vec.load("Data/Doc2Vec.model")


# In[97]:


Alpha = ["A", "B", "C", "D"]
QScores = []
Analysis = []
for ques in Questions:
    Q = ques["question"]["stem"]
    A = ques["question"]["choices"]
    C = ques["answerKey"]
    Scores = []
    for option in A:
        tempQ = Q + " " + option["text"]
        tempQ = SentPreProcessing(tempQ)
        tempQ = word_tokenize(tempQ)
        InferVec = model.infer_vector(tempQ)
        SimilarDoc = model.docvecs.most_similar([InferVec])
        maxi = FindMax(SimilarDoc)
        Scores.append(maxi)
    setOfOptions = []
    maxi = max(Scores)
    for i in range(len(Scores)):
        if maxi == Scores[i]:
            setOfOptions.append(Alpha[i])
    if (C in setOfOptions):
        QScores.append(1/len(setOfOptions))
    else:
        QScores.append(0)
    Analysis.append((maxi, C, setOfOptions))
    
print("Accuracy:", (sum(QScores)/len(Questions))*100)


# In[98]:


ShowAnalysis(Analysis, 20)
Counts(Analysis)


# In[ ]:




