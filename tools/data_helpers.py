# _*_ encoding:utf-8 _*_
__author__ = 'JQXX'
__date__ = '2018/6/13 6:48'
import numpy as np
import itertools
import codecs
from collections import Counter
import jieba


def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def load_data_and_labels(data_path,char_to_id_txt,max_len=0):
    read_file = open(char_to_id_txt, "r",encoding='utf-8')
    char_to_id={}
    for i in read_file:
        i=i.strip().split()
        if len(i)!=2:
            continue
        char_to_id[i[0]]=int(i[1])
    f = open(data_path, 'r', encoding='utf-8')
    y=[]
    line_sent=[]
    for line in f:
        line = line.strip().split()
        if len(line)!=2:
            continue
        if int(line[1]) == 1:
            y.append([0, 1])
        else:
            y.append([1, 0])
        temp_w=[]
        for i in line[0]:
            if i in char_to_id:
                temp_w.append(char_to_id[i])
            else:
                temp_w.append(0)
        line_sent.append(temp_w)
    if max_len==0:
        max_document_length = max([len(x) for x in line_sent])
    else:
        max_document_length=max_len
    text_x=[]
    for tmp in line_sent:
        if len(tmp)>=max_document_length:
            tmp=tmp[:max_document_length]
        while len(tmp)<max_document_length:
            tmp.append(0)
        text_x.append(tmp)
    return [text_x, y]


def load_test_data_and_labels(pos=None,neg=None):
    # Load data from files
    positive_examples = list(codecs.open("./data/test_text/pos.txt", "r", "utf-8").readlines())
    positive_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in positive_examples]
    negative_examples = list(codecs.open("./data/test_text/neg.txt", "r", "utf-8").readlines())
    negative_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def batch_iter(data, batch_size, num_epochs,is_shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if is_shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

