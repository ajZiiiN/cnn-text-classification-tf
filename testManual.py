#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime

# for stand-alone
import data_helpers
from text_cnn import TextCNN
#
# # for apis
# from cnn import data_helpers
# from cnn.text_cnn import TextCNN

from tensorflow.contrib import learn
import csv
import nltk
import sys

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("train_file", "../../data/2017-06-08/dataReplicated.csv", "Data source for the train data.")
tf.flags.DEFINE_string("test_file", "../../data/2017-06-07/dataReplicated.csv", "Data source for the test data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1498556243/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_manual", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def myTokenize (iter):
    for value in iter:
        yield nltk.word_tokenize(value)

# CHANGE THIS: Load data. Load your own data here
def getMsgAndPredict():
    n = int(input("Number of inputs.."))
    x_raw = []
    for i in range(n):
        x_raw.append(input())

    x_raw = np.array(x_raw)
    #print(x_raw)
    #Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.fit_transform(x_raw)))

    getPredictions(x_test)

def getMsgAndPredictApi(msg):
    x_raw = []
    x_raw.append(msg)

    x_raw = np.array(x_raw)
    #print(x_raw)
    #Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")

    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.fit_transform(x_raw)))

    return getPredictions(x_test)


# Evaluation
# ==================================================

def getPredictions(x_test):
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            print("Getting predictions...")
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            print("got Predictions...", all_predictions)

            return all_predictions

## for stand-alone
# getMsgAndPredict()