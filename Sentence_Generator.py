
import nltk
from nltk.corpus import brown
import collections as coll
import numpy as np
from nltk.util import ngrams

sent_list = []
for category in brown.categories():
    dummy = [sent for sent in brown.sents(categories = category)]
    for i in range(0,len(dummy)):
        dummy[i].insert(0,'<s>')
        dummy[i].insert(len(dummy[i]),'</s>')
    sent_list = sent_list + dummy
    
train_words = []
for i in range(0,len(sent_list)):
    for w in sent_list[i]:
        #if w.isalnum():
        train_words.append(w)
unigram = coll.Counter(train_words)

bgr = list(ngrams(train_words,2))
bigram_dict = coll.Counter(bgr)
tgr = list(ngrams(train_words,3))
trigram_dict = coll.Counter(tgr)



################ Calculating perplexity
def perp_measure(sent):
    N = 1/(len(sent) - 1)
    probab = (bigram_dict[(sent[0],sent[1])]/unigram[sent[0]])**N
    perp = 1/probab
    tgr = list(ngrams(sent,3))
    for i in range(len(tgr)):
        probab = trigram_dict[(tgr[i][0],tgr[i][1],tgr[i][2])]/bigram_dict[(tgr[i][0],tgr[i][1])]
        perp = perp*(1/((probab)**N))
    return perp



############### Finding the initial token
def initialize(sent):
    a = list(bigram_dict.keys())
    l = []
    for j in range(0,len(a)):
        if a[j][0] == sent[0]:
            l.append(a[j][1])
    while True:
        index = np.random.randint(len(l),size = 1)
        prob = bigram_dict[(sent[0],l[index[0]])]/unigram[sent[0]]
        if np.random.random_sample()<= prob:
            sent.append(l[index[0]])
            break
    return sent



end_char = ['.','?','!',';']



## Generating Sentences
iterations = 0
best_sent = []
perp_min = 500
while True:
    sent = ['<s>']
    sent = initialize(sent)
    a = list(trigram_dict.keys())            
    count = 1
    while count < 10:
        if count<9:
            l = []
            for j in range(0,len(a)):
                l1 = [a[j][0],a[j][1]]
                l2 = [sent[count-1],sent[count]]
                if l1 == l2 and a[j][2]!='</s>':
                    l.append(a[j][2])
            if len(l) == 0:
                count = 1
                sent = [sent[0],sent[1]]
                continue
            while True:
                index = np.random.randint(len(l),size = 1)
                prob = trigram_dict[(sent[count-1],sent[count],l[index[0]])]/bigram_dict[(sent[count-1],sent[count])]
                if np.random.random_sample()<= prob:
                    sent.append(l[index[0]])
                    count +=1
                    break
        else:
            l = []
            for j in range(0,len(a)):
                l1 = [a[j][0],a[j][1]]
                l2 = [sent[count-1],sent[count]]
                if l1 == l2 and a[j][2] in end_char:
                    l.append(a[j][2])
            if len(l) == 0:
                count = 1
                sent = [sent[0],sent[1]]
                continue
            while True:
                index = np.random.randint(len(l),size = 1)
                prob = trigram_dict[(sent[count-1],sent[count],l[index[0]])]/bigram_dict[(sent[count-1],sent[count])]
                if np.random.random_sample()<= prob:
                    sent.append(l[index[0]])
                    count +=1
                    break
    perp = perp_measure(sent)
    if perp < perp_min:
        perp_min = perp
        best_sent = sent
    if iterations==5 or perp_min<=5:
        break
    iterations += 1



gen_sent = best_sent[1]
for i in range(2,len(best_sent)):
    if best_sent[i] == '</s>':
        break
    gen_sent = gen_sent + ' ' + best_sent[i]
print(gen_sent)

