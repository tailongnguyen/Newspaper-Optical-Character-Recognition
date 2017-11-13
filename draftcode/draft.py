# -*- coding: UTF-8 -*-
from nltk.util import ngrams
import numpy as np 
import codecs
def extract_n_grams(sequence):
    for n in [2,4,6]:
        ngram = ngrams(sequence, n)
        # now you have an n-gram you can do what ever you want
        # yield ngram
        # you can count them for your language model?
        for item in ngram:
            lm[n][item] = lm[n].get(item, 0) + 1

def preprocess(string):
    # do what ever preprocessing that it needs to be done
    # e.g. convert to lowercase: string = string.lower()
    # return the sequence of tokens
    return string.split()

def postprocess(string):
    # do what ever preprocessing that it needs to be done
    # e.g. convert to lowercase: string = string.lower()
    # return the sequence of tokens
    if len(string) > 50:
        return None
    s = """0123456789abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ\%/.~,:-+()=><; "!?@*'[]{}^$#"""
    for c in string:
        if c not in s:
            return None
    return string

lm = {n:dict() for n in [2,4,6]}
print lm
with open('news.txt', 'r') as corpus:
    for line in corpus.readlines():
        sequence = preprocess(line)
        extract_n_grams(sequence)
        # print str(sequence)
num = 0
with codecs.open('grams1.txt', 'a', 'utf-8') as file:
    for k, v in lm.iteritems():
        print "%d-grams" % k
        for k1, v1 in v.iteritems():
            temp = postprocess(' '.join(k1))
            if temp != None:
                temp = unicode(temp, 'utf-8')
                choice = np.random.choice([0,1,2,3])
                if choice == 0:
                    temp = temp.upper()
                print temp ,v1
                file.write(temp+'\n')
                num+= 1
    print "total: %d" % num

# words_list = [ str(l).rstrip('\r\n') for l in codecs.open('grams.txt', 'r').readlines()]
# # assert len(words_list) == 48157
# # max_len = 1000
# # max_w = None
# with codecs.open('new-grams.txt', 'a', 'utf-8') as f:
#     for w in words_list:
#         if len(unicode(w, 'utf-8')) == 11:
#             print w
#             # max_len = len(unicode(w, 'utf-8'))
#             # max_w = w
#         choice = np.random.choice([0,1])
#         if choice == 0:
#             f.write(w.decode('utf-8') + '\n')
#         else:
#             f.write(w.decode('utf-8').upper() + '\n')
# print max_w
# print max_len