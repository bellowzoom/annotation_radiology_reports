import gensim
import logging
from gensim.models import Doc2Vec


def tr_doc():
    with open('/data/openi_all.txt','r') as oa:
        all_data=oa.read().split('\n')[:-1]
    indication=[]
    for line in all_data:
        indication.append(line.split('\t')[3])
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    documents = []
    count = 0
    for line in indication:
        words=line.split()
        documents.append(gensim.models.doc2vec.TaggedDocument(words, [str(count)]))
        count += 1
        if count % 10000 == 0:
            logging.info('{} has loaded...'.format(count))
    model = Doc2Vec(documents, dm=1, size=100, window=8, min_count=5, workers=4)
    model.save('ko_d2v.model')

def get_docvec(numb):
    return model.docvecs[numb]




