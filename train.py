import tensorflow as tf
import numpy as np
import datetime
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

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.data_path+"train.txt", FLAGS.data_path+"char_to_id.txt")


x = np.array(x_text)
y = np.array(y)
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y),dtype=np.int32))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]


dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))



with tf.Session() as sess:
    if FLAGS.choice_model=="rnn":
        cnn = Char_BiLSTM_CRF(x_train.shape[1], y_train.shape[1],
                          FLAGS.vocab_size,FLAGS.grad_clip,FLAGS.lr,FLAGS.embedding_dim, FLAGS.hidden_size,FLAGS.is_fine_tune)
    elif FLAGS.choice_model=="cnn":
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            vocab_size=FLAGS.vocab_size,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
    else:
        cnn = BiLSTM_cnn_model(x_train.shape[1], y_train.shape[1],
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
    if FLAGS.is_load_last_model:
        saver.restore(sess, FLAGS.model_save_path + "mybest")
    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        if FLAGS.choice_model != "cnn":
            _, loss, accuracy,leng = sess.run(
                [cnn.train_op, cnn.loss, cnn.accuracy,cnn.length],
                feed_dict)
        else:
            _, loss, accuracy = sess.run(
                [train_op, cnn.loss, cnn.accuracy],
                feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))


    def dev_step(x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 1.0
        }
        loss, accuracy = sess.run(
            [cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g}, dev_acc {:g}".format(time_str, loss, accuracy))
        return accuracy


    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    # Training loop. For each batch...
    ep=0
    best_acc=0
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        if ep%10==0:
            accuracy=dev_step(x_dev,y_dev)
            if accuracy>best_acc:
                path = saver.save(sess, FLAGS.model_save_path+"mybest")
                print("Saved model checkpoint to {}\n".format(path))
                best_acc=accuracy
        ep+=1
    accuracy = dev_step(x_dev, y_dev)
    if accuracy > best_acc:
        path = saver.save(sess, FLAGS.model_save_path+"mybest")
        print("Saved model checkpoint to {}\n".format(path))
        best_acc = accuracy
    batches = data_helpers.batch_iter(
        list(zip(x_dev, y_dev)), FLAGS.batch_size, FLAGS.num_epochs)
    # Training loop. For each batch...
    for batch in batches:
        try:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
        except:
            print("shit")
            continue
    path = saver.save(sess, FLAGS.model_save_path+"mybest")
    print("Saved model checkpoint to {}\n".format(path))
del cnn
tf.reset_default_graph()