

import nltk
import collections as coll
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.util import ngrams



def test_train_sent(gutenberg,brown):
    ############# Here all input data is tokenized in sentences and to each sentence start and end character are added.
    sent_list_g = []
    for file in gutenberg.fileids():
        dummy = [sent for sent in gutenberg.sents(file)]
        for i in range(0,len(dummy)):
            dummy[i].insert(0,'<s>')
            dummy[i].insert(len(dummy[i]),'</s>')
        sent_list_g = sent_list_g + dummy
        
    sent_list_b = []
    for category in brown.categories():
        dummy = [sent for sent in brown.sents(categories = category)]
        for i in range(0,len(dummy)):
            dummy[i].insert(0,'<s>')
            dummy[i].insert(len(dummy[i]),'</s>')
        sent_list_b = sent_list_b + dummy
    
    ############ Dividing the data in training, testing and development set.
    train_sent_g,test_sent_g = train_test_split(sent_list_g,test_size = 0.2,random_state = 4)
    train_sent_b,test_sent_b = train_test_split(sent_list_b,test_size = 0.2,random_state = 4)
    return train_sent_g,test_sent_g,train_sent_b,test_sent_b



def case_select(option,train_sent_g,test_sent_g,train_sent_b,test_sent_b):
    if option == 1:
        train_sent = train_sent_b
        test_sent = test_sent_b
    if option == 2:
        train_sent = train_sent_g
        test_sent = test_sent_g
    if option == 3:
        train_sent = train_sent_b + train_sent_g
        test_sent = test_sent_b
    if option == 4:
        train_sent = train_sent_g + train_sent_b
        test_sent = test_sent_g
    return train_sent,test_sent



def data_prep(option,train_sent_g,test_sent_g,train_sent_b,test_sent_b):
    
    train_sent,test_sent = case_select(option,train_sent_g,test_sent_g,train_sent_b,test_sent_b)    
    train_words = []
    test_words = []
    count_t = 0
    for i in range(0,len(train_sent)):
        for w in train_sent[i]:
            train_words.append(w)
    for i in range(0,len(test_sent)):
        for j in range(0,len(test_sent[i])):
            test_words.append(test_sent[i][j])
            if test_sent[i][j] == '<s>':
                count_t += 1
    N_t = len(test_words) - count_t
    return train_words,test_words,N_t


def data_prep2_Katz(train_words,test_words,K,d):
    ############# This block generates some <UNK> in the training data and generate unigram and bigram dictionaries
    unigram = coll.Counter(train_words)
    count = 0
    for i in range(0,len(train_words)):
        if unigram[train_words[i]] == 1 and count<=K:
            train_words[i] = '<UNK>'
            count += 1
    unigram_new = coll.Counter(train_words)
    bgr = list(ngrams(train_words,2))
    bigram_dict = coll.Counter(bgr)
    del bigram_dict[('</s>','<s>')] #### This bigram will only artifically decrease the perplexity
    
    ################# Calculating the parameters alpha and beta for each unigram
    alpha_w = {}
    beta_w = {}
    a = list(unigram_new.keys())
    for i in range(len(a)):
        alpha_w[a[i]] = 0
        beta_w[a[i]] = sum(unigram_new.values())
    a = list(bigram_dict.keys())
    for i in range(len(a)):
        alpha_w[a[i][0]] = alpha_w[a[i][0]] + (bigram_dict[a[i]]- d)
        beta_w[a[i][0]] = beta_w[a[i][0]] - unigram_new[a[i][1]]
        
    ################ Preparing test data for perplexity measurement.
    for i in range(0,len(test_words)):
        if test_words[i] not in unigram_new.keys():
            test_words[i] = '<UNK>'
    return unigram_new, bigram_dict,alpha_w,beta_w,test_words

	
def data_prep2_Kn(train_words,test_words,K,d):
    ############# This block generates some <UNK> in the training data and generate unigram and bigram dictionaries
    unigram = coll.Counter(train_words)
    count = 0
    for i in range(0,len(train_words)):
        if unigram[train_words[i]] == 1 and count<=K:
            train_words[i] = '<UNK>'
            count += 1
    unigram_new = coll.Counter(train_words)
    bgr = list(ngrams(train_words,2))
    bigram_dict = coll.Counter(bgr)
    del bigram_dict[('</s>','<s>')] #### This bigram will only artifically decrease the perplexity
    
    ################# Calculating the continuation probability of each unigram
    cont_word = []
    a = list(bigram_dict.keys())
    for i in range(0,len(a)):
        cont_word.append(a[i][1])
    P_cont = coll.Counter(cont_word)

    ################# Calculting the normalizing factor lambda for each unigram
    discount = d
    first_ = []
    for i in range(0,len(a)):
        first_.append(a[i][0])
    first_dict = coll.Counter(first_)
    first_word = list(first_dict.keys())
    lambda_w = {}
    for i in range(0,len(first_word)):
        val = (discount*first_dict[first_word[i]])/unigram_new[first_word[i]]
        lambda_w[first_word[i]] = val
        
    ################ Preparing test data for perplexity measurement.
    for i in range(0,len(test_words)):
        if test_words[i] not in unigram_new.keys():
            test_words[i] = '<UNK>'
    return unigram_new, bigram_dict,P_cont,lambda_w,test_words

	
def data_prep2_Sb(train_words,test_words,K,d):
    ############# This block generates some <UNK> in the training data and generate unigram, bigram and trigram dictionaries
    unigram = coll.Counter(train_words)
    count = 0
    for i in range(0,len(train_words)):
        if unigram[train_words[i]] == 1 and count<=K:
            train_words[i] = '<UNK>'
            count += 1
    unigram_new = coll.Counter(train_words)
    bgr = list(ngrams(train_words,2))
    bigram_dict = coll.Counter(bgr)
    tgr = list(ngrams(train_words,3))
    trigram_dict = coll.Counter(tgr)
        
    ################ Preparing test data for perplexity measurement.
    for i in range(0,len(test_words)):
        if test_words[i] not in unigram_new.keys():
            test_words[i] = '<UNK>'
    return unigram_new, bigram_dict,trigram_dict,test_words


def perp_score_Katz(unigram_new, bigram_dict,alpha_w,beta_w,test_words,d,N):
    perp = 1
    for i in range(1,len(test_words)):
        if test_words[i] == '<s>' and test_words[i-1] == '</s>':
            probab = 1
        elif bigram_dict[(test_words[i-1],test_words[i])]>0:
            probab = (bigram_dict[(test_words[i-1],test_words[i])] - d)/unigram_new[test_words[i-1]]
        else: 
            probab = (1 - (alpha_w[test_words[i-1]]/unigram_new[test_words[i-1]]))*(unigram_new[test_words[i]]/beta_w[test_words[i-1]])
        perp = perp*((1/probab)**N)
    return perp

	
def perp_score_Kn(unigram_new, bigram_dict,P_cont,lambda_w,test_words,d,N):
    perp = 1
    for i in range(1,len(test_words)):
        if test_words[i] == '<s>' and test_words[i-1] == '</s>':
            probab = 1
        else: 
            probab = (max(bigram_dict[(test_words[i-1],test_words[i])] - d,0))/unigram_new[test_words[i-1]] + lambda_w[test_words[i-1]]*(P_cont[test_words[i]]/len(bigram_dict.keys()))
            #print(probab)
        perp = perp*((1/probab)**N)
    return perp


def perp_score_Sb(unigram_new, bigram_dict,trigram_dict,test_words,d,N):
    perp = (bigram_dict[(test_words[0],test_words[1])]/unigram_new[test_words[0]])**N
    for i in range(2,len(test_words)):
        if trigram_dict[(test_words[i-2],test_words[i-1],test_words[i])]>0:
            probab = trigram_dict[(test_words[i-2],test_words[i-1],test_words[i])]/bigram_dict[(test_words[i-2],test_words[i-1])]
        elif bigram_dict[(test_words[i-1],test_words[i])]>0:
            probab = d*(bigram_dict[(test_words[i-1],test_words[i])]/unigram_new[test_words[i-1]])
        else: 
            probab = d*d*(unigram_new[test_words[i]])/sum(unigram_new.values())
        perp = perp*((1/probab)**N)
    return perp




def performance_measure_Katz(train_words,test_words,opt_disc,N_t,K):
    unigram_new, bigram_dict,alpha_w,beta_w,test_words = data_prep2_Katz(train_words,test_words,K,opt_disc)
    perp = perp_score_Katz(unigram_new, bigram_dict,alpha_w,beta_w,test_words,opt_disc,1/N_t)
    return perp




def performance_measure_Kn(train_words,test_words,opt_disc,N_t,K):
    unigram_new, bigram_dict,P_cont,lambda_w,test_words = data_prep2_Kn(train_words,test_words,K,opt_disc)
    perp = perp_score_Kn(unigram_new, bigram_dict,P_cont,lambda_w,test_words,opt_disc,1/N_t)
    return perp



def performance_measure_Sb(train_words,test_words,opt_disc,N_t,K):
    unigram_new, bigram_dict,trigram_dict,test_words = data_prep2_Sb(train_words,test_words,K,opt_disc)
    perp = perp_score_Sb(unigram_new, bigram_dict,trigram_dict,test_words,opt_disc,1/N_t)
    return perp




################### MAIN #####################

from nltk.corpus import gutenberg
from nltk.corpus import brown

########## These values of the hyperparameter has been tuned in a different experiment.

opt_disc_Katz = 0.8
opt_disc_Kn = 0.85
opt_disc_Sb = 0.7

for i in range(1,5):
    option = i
    K = 5000 ## No. of words to be converted as <'UNK'>.
    train_sent_g,test_sent_g,train_sent_b,test_sent_b = test_train_sent(gutenberg,brown)
    train_words,test_words,N_t = data_prep(option,train_sent_g,test_sent_g,train_sent_b,test_sent_b)
    test_perp_Katz = performance_measure_Katz(train_words,test_words,opt_disc_Katz,N_t,K)
    test_perp_Kn = performance_measure_Kn(train_words,test_words,opt_disc_Kn,N_t,K)
    test_perp_Sb = performance_measure_Sb(train_words,test_words,opt_disc_Sb,N_t,K)
    print('Perplexity Values for Setting S',option)
    print('Bigram Katz: ',test_perp_Katz,'Bigram Kneser-Ney: ',test_perp_Kn,'Trigram StupidBackOff: ',test_perp_Sb)




########## Sentence Generation Code

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

end_char = ['.','?','!',';']

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
    if iterations==10 or perp_min<=5:
        break
    iterations += 1
    
gen_sent = best_sent[1]
for i in range(2,len(best_sent)):
    if best_sent[i] == '</s>':
        break
    gen_sent = gen_sent + ' ' + best_sent[i]
print(gen_sent)

