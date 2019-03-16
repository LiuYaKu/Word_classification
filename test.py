# _*_ encoding:utf-8 _*_
__author__ = 'JQXX'
__date__ = '2018/11/3 15:52'
import tensorflow as tf
import numpy as np
import jieba
from tools import data_helpers
from cnn_network import TextCNN
from rnn_network import Char_BiLSTM_CRF
from rnn_cnn_model import BiLSTM_cnn_model
import pickle
from config import FLAGS
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if FLAGS.is_use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "20000000"

test_file_name=FLAGS.test_data_path.split("/")[-1]

if not FLAGS.is_domain:
    test_file_name=test_file_name[2:]
res=open(FLAGS.result_path+test_file_name+"_result.txt","w",encoding='utf-8')
f_res=open(FLAGS.result_path+test_file_name+"_f_result.txt","w",encoding='utf-8')
m=open(FLAGS.test_data_path,"r",encoding="utf-8")

test_data=set()
for i in m:
    line=i.strip()
    test_data.add(line)

test_data=list(test_data)

read_file = open(FLAGS.data_path+"char_to_id.txt", "r",encoding='utf-8')
char_to_id={}
for i in read_file:
    i=i.strip().split()
    if len(i)!=2:
        continue
    char_to_id[i[0]]=int(i[1])
line_sent=[]
for line in test_data:
    temp_w=[]
    for i in line:
        if i in char_to_id:
            temp_w.append(char_to_id[i])
        else:
            temp_w.append(0)
    line_sent.append(temp_w)


print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.data_path+"train.txt", FLAGS.data_path+"char_to_id.txt")

x = np.array(x_text)
y = np.array(y)

max_document_length = x.shape[1]

text_x=[]
for tmp in line_sent:
    if len(tmp)>=max_document_length:
        tmp=tmp[:max_document_length]
    while len(tmp)<max_document_length:
        tmp.append(0)
    text_x.append(tmp)
text_x=np.array(text_x)
test_w=np.array(test_data)


with tf.Session() as sess:
    if FLAGS.choice_model=="rnn":
        cnn = Char_BiLSTM_CRF(x.shape[1], y.shape[1],
                          FLAGS.vocab_size,FLAGS.grad_clip,FLAGS.lr,FLAGS.embedding_dim, FLAGS.hidden_size,FLAGS.is_fine_tune)
    elif FLAGS.choice_model == "cnn":
        cnn = TextCNN(
            sequence_length=x.shape[1],
            num_classes=y.shape[1],
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            vocab_size=FLAGS.vocab_size,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
    else:
        cnn = BiLSTM_cnn_model(x.shape[1], y.shape[1],
                          FLAGS.vocab_size,FLAGS.grad_clip,FLAGS.lr,list(map(int, FLAGS.filter_sizes.split(","))),
                         FLAGS.num_filters,FLAGS.embedding_dim, FLAGS.hidden_size,FLAGS.is_fine_tune)

    saver = tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())
    ce = np.random.uniform(-1, 1, [FLAGS.vocab_size + 1, FLAGS.embedding_dim])
    word2vec_model = pickle.load(open(FLAGS.data_path+"char2vec_xinchou_"+str(FLAGS.embedding_dim)+".pkl", 'rb'))
    f = open(FLAGS.data_path+"char_to_id.txt", "r", encoding='utf-8')
    ce[0] = np.zeros(FLAGS.embedding_dim)
    for i in f:
        i = i.strip().split()
        try:
            ce[int(i[1])] = word2vec_model[i[0]]
        except:
            print(i[0])
    sess.run(cnn.char_embedding_init, feed_dict={cnn.char_embedding_placeholder: ce})
    def dev_step(x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.dropout_keep_prob: 1.0
        }
        predictions = sess.run(
            [cnn.soft_scores],
            feed_dict)
        predictions=predictions[0]
        for i in range(len(y_batch)):
            if predictions[i][1]>FLAGS.threshold:
                is_ok = True
                for w in jieba.lcut(y_batch[i]):
                    if len(w) == 1:
                        f_res.write(y_batch[i] + "\n")
                        is_ok = False
                        break
                if is_ok:
                    res.write(y_batch[i] + "\n")
            else:
                f_res.write(y_batch[i]+"\n")
    saver.restore(sess, FLAGS.model_save_path+'mybest')
    batches = data_helpers.batch_iter(
        list(zip(text_x, test_w)),FLAGS.batch_size,1,is_shuffle=False)
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        dev_step(x_batch, y_batch)
res.close()
m.close()
f_res.close()
del cnn
tf.reset_default_graph()