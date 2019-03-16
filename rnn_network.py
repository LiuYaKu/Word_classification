from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np

class Char_BiLSTM_CRF(object):
    def __init__(self, sequence_length, num_classes,
          char_vocab_size,grad_clip,learning_rate,char_embedd_dim = 30, n_hidden_LSTM =200,is_fine_tune=True,num_layers=2):
        self.sentence_length = sequence_length
        self.rnn_size = n_hidden_LSTM
        self.class_size = num_classes
        self.num_layers=num_layers
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length])
        self.vocab_size=char_vocab_size
        self.char_embedding_placeholder = tf.placeholder(tf.float32, [char_vocab_size+1, char_embedd_dim])
        self.lr = learning_rate
        self.input_y = tf.placeholder(tf.float32, [None, num_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        words_used_in_sent = tf.sign(tf.abs(self.input_x))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, axis=1), tf.int32)

        with tf.device('/cpu:0'), tf.name_scope("char_embedding"):
            self.W_char = tf.Variable(tf.random_uniform([char_vocab_size+1, char_embedd_dim],-1,1),trainable=is_fine_tune, name="W_char")
            self.embedded_char = tf.nn.embedding_lookup(self.W_char, self.input_x, name="embedded_char")
            self.char_embedding_init = self.W_char.assign(self.char_embedding_placeholder)

        self.add_logits_op()

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.soft_scores = tf.nn.softmax(self.prediction)
        self.re_prediction = tf.cast(tf.argmax(self.prediction, 1), tf.int32,
                                     name="output_node")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.re_prediction,tf.cast(tf.argmax(self.input_y, 1), tf.int32)),tf.float32))

    def add_logits_op(self):
        with tf.variable_scope("bi-lstm",reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(self.rnn_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.rnn_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.embedded_char,
                    sequence_length=self.length, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout_keep_prob)

        with tf.variable_scope("proj",reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.rnn_size*self.sentence_length, self.class_size])

            b = tf.get_variable("b", shape=[self.class_size],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            output = tf.reshape(output, [-1, 2*self.rnn_size*self.sentence_length])
            self.prediction = tf.matmul(output, W) + b