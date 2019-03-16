from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
import config

class BiLSTM_cnn_model(object):
    def __init__(self, sequence_length, num_classes,
          char_vocab_size,grad_clip,learning_rate,filter_sizes,num_filters,char_embedd_dim = 50, n_hidden_LSTM =200,is_fine_tune=True):
        self.sentence_length = sequence_length
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.rnn_size = n_hidden_LSTM
        self.embedding_size=char_embedd_dim
        self.class_size = num_classes
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
            rnn_output = tf.concat([output_fw, output_bw], axis=-1)
            rnn_output = tf.nn.dropout(rnn_output, self.dropout_keep_prob)
            rnn_output = tf.expand_dims(rnn_output, -1)

        with tf.name_scope("cnn_layer"):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.rnn_size*2, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        rnn_output,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.sentence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.num_filters * len(self.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.variable_scope("proj",reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[num_filters_total, self.class_size])

            b = tf.get_variable("b", shape=[self.class_size],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            output = tf.reshape(self.h_drop,[-1, num_filters_total])
            self.prediction = tf.matmul(output, W) + b

# if __name__=="__main__":
#     BiLSTM_cnn_model(55, 2,5000,10,0.01,[3,4,5],128,50,200,False)