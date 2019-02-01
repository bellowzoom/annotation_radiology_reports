# -*- coding: utf-8 -*-
import sys
import logging
import os
import gensim
# 引入doc2vec
from gensim.models import Doc2Vec

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utilties import ko_title2words

documents = []
count = 0
with open('../data/titles/ko.video.corpus', 'r') as f:
    for line in f:
        title = unicode(line, 'utf-8')
        words = ko_title2words(title)
        documents.append(gensim.models.doc2vec.TaggedDocument(words, [str(count)]))
        count += 1
        if count % 10000 == 0:
            logging.info('{} has loaded...'.format(count))

model = Doc2Vec(documents, dm=1, size=100, window=8, min_count=5, workers=4)
model.save('models/ko_d2v.model')
