# _*_ encoding:utf-8 _*_
__author__ = 'JQXX'
__date__ = '2018/10/25 18:54'
from gensim.models.word2vec import Word2Vec
import re
import pickle
import os
os.chdir('../')
from config import FLAGS
all_sen=[]
files=os.listdir(FLAGS.data_path+"book/")

for path in files:
    f=open(FLAGS.data_path+"book/"+path,"r",encoding='utf-8')
    paragraph = "".join([m for m in f.readlines() if m.strip()])
    sentences = re.split('(。|！|\!|\.|？|\?)', paragraph)

    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        sent = sent.replace("\n", "")
        sent = sent.replace("\t", "")
        sent = sent.replace(" ", "")
        all_sen.append([c for c in sent if c.strip()])


word2vec = Word2Vec(all_sen, size=FLAGS.embedding_dim, window=5,
                                       workers=3,
                                       sg=1,
                                       batch_words=10000, min_count=1)

pickle.dump(word2vec,open(FLAGS.data_path+"char2vec_xinchou_"+str(FLAGS.embedding_dim)+".pkl",'wb'))