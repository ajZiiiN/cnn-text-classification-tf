
from __future__ import print_function

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

import os
import sys
from tensorflow.contrib import learn
import numpy as np
import nltk

# TensorFlow serving stuff to send messages
sys.path.append("/home/zi/sandbox/tf_serving_example")
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1498556243/checkpoints/", "Checkpoint directory from training run")



# Command line arguments
tf.app.flags.DEFINE_string('server', '172.17.0.2:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def myTokenize (iter):
    for value in iter:
        yield nltk.word_tokenize(value)


def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request

    # See prediction_service.proto for gRPC request/response details.
    data = ["How are you"]
    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.fit_transform(data)))


    request = predict_pb2.PredictRequest()
    print("Got request object... ")
    # Call GAN model to make prediction on the image
    request.model_spec.name = 'abuse'
    request.model_spec.signature_name = 'predict_text'
    request.inputs['text'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_test[0].astype(np.int32), shape=[1, x_test[0].size]))
    print("Copied input to request..")
    result = stub.Predict(request, 60.0)  # 60 secs timeout
    print(result)


if __name__ == '__main__':
    tf.app.run()