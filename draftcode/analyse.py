# -*- coding: UTF-8 -*-
import codecs
import matplotlib.pyplot as plt

s = """0123456789abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ\%/.~,:-+()=><; "!?@*'[]{}^$#"""
hist = {}
for i, c in enumerate(unicode(s, 'utf-8')):
	hist[c] = 0
print len(hist)
words_list = [ str(l).rstrip('\r\n') for l in open('grams.txt', 'r').readlines()]

for word in words_list:
    for c in unicode(word, 'utf-8'):
        try:
            hist[c] +=1
        except KeyError:
            continue

# for k, v in hist.iteritems():
    # print k, v
plt.bar(range(225), hist.values(), width=1.0)
plt.show()