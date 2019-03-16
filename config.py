# _*_ encoding:utf-8 _*_
__author__ = 'JQXX'
__date__ = '2018/11/13 16:34'
import tensorflow as tf

tf.flags.DEFINE_float("dev_sample_percentage", 0.01, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("vocab_size",6000, "Batch Size (default: 64)")

tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob",0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("grad_clip", 10, "grad_clip")
tf.flags.DEFINE_float("lr",1e-3, "lr")

tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs",50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("hidden_size",64, "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_string("data_path","./data/", "data path")
tf.flags.DEFINE_string("result_path","./result/", "result path")
tf.flags.DEFINE_string("model_save_path","./model/best_model/", "model save path")
tf.flags.DEFINE_string("test_data_path","./data/test.txt", "test data path")
tf.flags.DEFINE_float("threshold",0.9, "The larger the threshold, the better the result, but the smaller the number of words, and the maximum value is 1")
tf.flags.DEFINE_bool("is_use_gpu",True, "Whether GPU is used")
tf.flags.DEFINE_string("gpu_id","1", "Which GPU to use")
tf.flags.DEFINE_bool("is_domain",True,"is domain")
tf.flags.DEFINE_string("choice_model","cnn","rnn cnn rnn_cnn")
tf.flags.DEFINE_bool("is_fine_tune",False,"is fine tune embedding layer")
tf.flags.DEFINE_bool("is_load_last_model",True,"is load last model")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
if not FLAGS.is_domain:
    FLAGS.data_path = FLAGS.data_path + "z_"
    FLAGS.result_path = FLAGS.result_path + "z_"
    FLAGS.model_save_path = "./model/z_best_model/"
    if FLAGS.choice_model!="cnn":
        FLAGS.result_path = FLAGS.result_path + FLAGS.choice_model + "_"
        FLAGS.model_save_path = "./model/z_"+ FLAGS.choice_model +"_best_model/"
elif FLAGS.choice_model!="cnn":
    FLAGS.result_path = FLAGS.result_path + FLAGS.choice_model + "_"
    FLAGS.model_save_path = "./model/" + FLAGS.choice_model + "_best_model/"

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
