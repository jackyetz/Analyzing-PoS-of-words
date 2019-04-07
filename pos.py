#!/usr/bin/env python
# encoding: utf-8
import nltk
import numpy as np
from collections import Counter
txtlines = []
vocab = {}
tokenlist = [] # token list
poslist = [] # pos list aligning token in tokenlist
with open('./dialogues_text.txt','r',encoding='utf-8') as fo:
    for line in fo.read().splitlines():
        line = line.lower()
        txttmp = [s.strip() for s in line.split("__eou__")[:-1]]
        txtlines.append(' '.join(txttmp))
for line in txtlines:
    pos_tags = nltk.pos_tag(nltk.word_tokenize(line))
    for word, flag in pos_tags:
        if word.isspace(): continue
        tokenlist.append(word)
        poslist.append(flag)
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

numPoSChanged_tokens = 0 # number of PoS-changed words
numPoSChanged_times = 0  # number of PoS-changed times
listChanged_tokens = {} # list of PoS-changed words
for voctoken in vocab.keys():
    indexlist = [i for i, token in enumerate(tokenlist) if token == voctoken]
    possublist = np.array(poslist)[indexlist]
    resultdic = Counter(possublist)
    resultlist = list(resultdic.values())
    if(len(resultlist)>1):
        numPoSChanged_tokens += 1
        numPoSChanged_times += sum(resultlist)-max(resultlist)
        listChanged_tokens[voctoken] = resultdic
    if (False):  # diagnose
        print(indexlist)
        print(possublist)
        tmpstr = ''
        for i in range(50):
            tmpstr += '(' + tokenlist[i] + ',' + poslist[i] + ')'
        print(tmpstr)
        exit()
printline = 'tokens: %d, vocab: %d, numPoSChanged_tokens: %d, numPoSChanged_times: %d. percent of changedTokens: %f, percent of changedTimes: %f' % (len(tokenlist),len(vocab),numPoSChanged_tokens,numPoSChanged_times,float(numPoSChanged_tokens)/len(vocab),float(numPoSChanged_times)/len(tokenlist))
print(printline)
# write to file
with open('result.txt','w',encoding='utf-8') as wp:
    wp.writelines(printline + '\n')
    for (key,value) in listChanged_tokens.items():
        wp.writelines(key+':::'+str(value)+'\n')