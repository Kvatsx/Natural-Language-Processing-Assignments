#!/usr/bin/env python
# coding: utf-8

# ### Kaustav Vats (2016048)

# In[157]:


import numpy as np
import nltk
import json
import copy


# ### Load Grammar, Sentences

# In[158]:


# Load Grammer
Grammar = nltk.data.load("grammars/large_grammars/atis.cfg")
# Grammar = nltk.CFG.fromstring("""
# A -> B C D
# B -> E F | G
# C -> 'c'
# D -> C B G
# E -> F G
# E -> 'e'
# F -> 'f'
# G -> 'g'
# """)
old_productions = Grammar.productions()

# Load Raw Sentences
Sentences = nltk.data.load("grammars/large_grammars/atis_sentences.txt")


# ### All required functions

# In[189]:


class Node:
    def __init__(self, root, left, right, val):
        self.root = root
        self.left = left
        self.right = right
        self.val = val
        
    def __str__(self):
        return self.root
        
    def __repr__(self):
        return self.__str__()

def save(dic):
    with open('productions.json', 'w') as fp:
        json.dump(dic, fp)

def pre_process_grammar(prod):
    nprod = {}
    for i in range(len(prod)):
        k = str(prod[i].lhs())
        v = list(prod[i].rhs())
#         print(v)
        
        for i in range(len(v)):
            if isinstance(v[i], str) or isinstance(v[i], bytes):
                v[i] = "'" + v[i] + "'"
            else:
                v[i] = str(v[i])
        
        if k not in nprod:
            nprod[k] = [v]
        else:
            if v not in nprod[k]:
                 nprod[k].append(v)
    save(nprod)
    return nprod

def match(dic, label, x, y):
    for i in range(len(dic)):
        if dic[i][0] == x and dic[i][1] == y:
            return i
    return -1

def RemoveLargeRules(old_prod):
    count = 0
    dic = []
    label = []
    temp_prod = copy.deepcopy(old_prod)
    for k in temp_prod:
        v = temp_prod[k]
        for idx in range(len(v)):
            if (len(v[idx]) > 2):  # rule with more than 2 non terminal
                prev = v[idx][0]
                for i in range(1, len(v[idx])-1):
                    curr = v[idx][i]
                    index = match(dic, label, prev, curr)
                    if (index == -1):
                        letter = "NR" + str(count)
                        count += 1
                        old_prod[letter] = [[prev, curr]]
                        dic.append([prev, curr])
                        label.append(letter)
                        prev = letter
                    else:
                        letter = label[index]
                        prev = letter
                old_prod[k][idx] = [prev, v[idx][len(v[idx])-1]]
                        
    return old_prod
    
def RemoveUnitRules(old_prod):
    ustate = list(old_prod.keys())
    flag = True
    while flag:
        flag = False
        temp_productions = copy.deepcopy(old_prod)
        for k in temp_productions:
            v = temp_productions[k]
            for idx in range(len(v)):
                if (len(v[idx]) == 1 and v[idx][0] in ustate and v[idx][0] != k):
                    flag = True
                    k2 = copy.deepcopy(v[idx][0])
                    v2 = old_prod[k2]
                    old_prod[k].remove(v[idx])
                    for j in range(len(v2)):
                        old_prod[k].append(v2[j])
    return old_prod

def RemoveDuplicate(old_prod):
    temp_prod = copy.deepcopy(old_prod)
    for k in temp_prod:
        v = temp_prod[k]
        new_v = []
        for e in v:
            if e not in new_v:
                new_v.append(e)
            else:
                old_prod[k].remove(e)
    return old_prod

def save_CNF(old_prod):
    with open("CNF.txt", 'w') as fp:
        for k in old_prod:
            v = old_prod[k]
            res = ""
            for idx in range(len(v)):
                res = ""
                for i in range(len(v[idx])):
                    res += str(v[idx][i]) + " "
                fp.write(str(k) + " -> " + res + "\n")
        
    
"""
Assumptions and Observations
1. Eliminate start symbol from RHS. = This case is not present in given grammar, no such grammar would be given which has this case.
2. Eliminate null = This case is not present in given grammar, no such grammar would be given which has this case.
3. Eliminate terminals from RHS if they exist with other terminals or non-terminals = This case is not present in given grammar, no such grammar would be given which has this case.
4. 
"""
def CFG2CNF(old_prod):
    old_prod = pre_process_grammar(old_prod)
    old_prod = RemoveUnitRules(old_prod)
    
#     old_prod = large(old_prod)
    old_prod = RemoveLargeRules(old_prod)
    old_prod = RemoveDuplicate(old_prod)
    save_CNF(old_prod)
    return old_prod

def PreProcess(old_prod):
    old_prod = pre_process_grammar(old_prod)
    save_CNF(old_prod)
    return old_prod
    
def cky_parser(sent, old_prod):
    sentence = []
    for i in range(len(sent)):
        sentence.append("'" + sent[i] + "'")
    DP = []
    NodeMat = []
    for i in range(len(sentence)):
        DP.append([])
        NodeMat.append([])
        for j in range(len(sentence)):
            DP[i].append([])
            NodeMat[i].append([])
            
    # Bottom Up approach to fill Mat
    for i in range(1, len(sentence)):
        for k in old_prod:
            v = old_prod[k]
            for rhs in v:
                if (len(rhs) == 1 and rhs[0] == sentence[i-1]):
                    DP[i-1][i].append(k)
                    NodeMat[i-1][i].append(Node(k, None, None, sentence[i-1]))
                    
        for j in range(i-1, -1, -1):
            for k in range(j+1, i):
                for key in old_prod:
                    v = old_prod[key]
                    for idx in range(len(v)):
                        if (len(v[idx]) == 2):  # A = BC
                            B = v[idx][0]
                            C = v[idx][1]
                            if B in DP[j][k] and C in DP[k][i]:
                                DP[j][i].append(key)
                                for b in NodeMat[j][k]:
                                    for c in NodeMat[k][i]:
                                        if b.root == B and c.root == C:
                                            NodeMat[j][i].append(Node(key, b, c, None))
                                            
    return NodeMat[0][len(sentence)-1], NodeMat

# def CKY_Parser(sent, old_prod):
#     n = len(sent)
#     sentence = []
#     for i in range(n):
#         sentence.append("'" + sent[i] + "'")
#     # Initialization step
#     Mat = []
#     DP = []
#     for i in range(n):
#         Mat.append([])
#         DP.append([])
#         for j in range(n):
#             Mat[i].append([])
#             DP[i].append([])
    
#     # For Variables with 1 substring
#     for i in range(n):
#         for key in old_prod:
#             val = old_prod[key]
#             for rhs in val:
#                 if (len(rhs) == 1 and rhs[0] == sentence[i]):
#                     Mat[i-1][i].append(key)
#                     DP[i-1][i].append(Node(key, None, None, sentence[i]))
                    
#     for i in range(len(Mat)):
#         print()
#         for j in range(len(Mat[i])):
#             print(Mat[i][j], end=' ')
    
#     # For rest of the variables
#     for j in range(2, n):
#         for i in range(j-2, -1, -1):
#             for k in range(i+1, j-2):
#                 for key in old_prod:
#                     val = old_prod[key]
#                     for idx in range(len(val)):
#                         if len(val[idx]) == 2:  # A = BC
#                             B = val[idx][0]
#                             C = val[idx][1]
#                             if B in Mat[i][k] and C in Mat[k][j]:
#                                 Mat[i][j].append(key)
#                                 for b in DP[i][k]:
#                                     for c in DP[k][j]:
#                                         if b.root == B and c.root == C:
#                                             DP[i][j].append(Node(key, b, c, None))

#     return DP[0][n-1], DP

def ShowTree(root):
    if root.val != None:
        return "(" + root.root + " " + root.val + ")"
    left = 2 + len(root.left.root)
    right = 2 + len(root.right.root)
    return "(" + root.root + " " + ShowTree(root.left) + " " +  ShowTree(root.right) + ")"
                


# ### This step converts production rules to CNF form with some preconditions and assumptions, mentioned with the 

# In[190]:


# old_prod = pre_process_grammar(old_productions)
# old_prod = CFG2CNF(old_productions)
G = Grammar.chomsky_normal_form(new_token_padding="_")
old_productions = G.productions()
old_prod = PreProcess(old_productions)


# ### Load Test Sentences

# In[191]:


s = nltk.data.load("grammars/large_grammars/atis_sentences.txt")
t = nltk.parse.util.extract_test_sentences(s)
sentences = []
for sent in t:
    sentences.append(sent[0])


# ### Part - 1  |  Below prints the CKY parser count for all test sentences

# In[192]:


Ans = []
start = str(Grammar.start())
for i in range(len(sentences)):
# i = 2
    bt, DP = cky_parser(sentences[i], old_prod) 
    ntree = 0

    #     for i in range(len(DP)):
    #         print()
    #         for j in range(len(DP[i])):
    #             print(DP[i][j], end=' ')
    for node in bt:
        if node.root == start:
            ntree += 1
    Ans.append(str(ntree))

    # OriginalParse Count and My parser parse Count
    print(str(t[i][1]) + " " + str(ntree))
        


# ### Part 2   |   Print Tree and Draw them using nltk.draw

# In[173]:


# for i in range(len(Ans)):
#     print(i, Ans[i])
bt, DP = cky_parser(sentences[91], old_prod) 
Trees = []
for node in bt:
    if node.root == start:
        ntree += 1
        tr = ShowTree(node)
        Trees.append(tr)
        print(tr)
        print("\n")


# In[175]:


# Run this to draw trees
for tr in Trees:
    tree = nltk.Tree.fromstring(tr)
    tree.draw()


# In[176]:


print(sentences[91])


# In[ ]:




