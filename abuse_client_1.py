# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf
from tensorflow.contrib import learn

import sys
import nltk
import os
sys.path.append("/home/zi/sandbox/tf_serving_example")
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from datetime import datetime as dt
from time import sleep

tf.app.flags.DEFINE_integer('concurrency', 20,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 1000, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1498556243/checkpoints/", "Checkpoint directory from training run")

FLAGS = tf.app.flags.FLAGS

def myTokenize (iter):
    for value in iter:
        yield nltk.word_tokenize(value)


class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()
        self._res_time = 0
        self._res_time_li = []

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

    def get_response_time(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._res_time / self._num_tests

    def add_response_time(self, sec):
        with self._condition:
            self._res_time += sec
            self._condition.notify()

def _create_rpc_callback(label, result_counter, start_time):
    """Creates RPC callback function.

    Args:
      label: The correct label for the predicted example.
      result_counter: Counter for the prediction result.
    Returns:
      The callback function.
    """
    def _callback(result_future):
        """Callback function.

        Calculates the statistics for the prediction result.

        Args:
          result_future: Result future of the RPC.
        """
        #print("recieved by callback")
        total_sec = (dt.now() - start_time).seconds
        result_counter.add_response_time(total_sec)
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            #print("no exception found.....")
            sys.stdout.write('.')
            sys.stdout.flush()
            response = numpy.array(
                result_future.outputs['scores'].int64_val)
            #prediction = numpy.argmax(response)
            #print("Prediction: ", response)
            # if label != prediction:
            #     result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback


def do_inference(hostport, work_dir, concurrency, num_tests, vocab_processor):
    """Tests PredictionService with concurrent requests.

    Args:
      hostport: Host:port address of the PredictionService.
      work_dir: The full path of working directory for test data set.
      concurrency: Maximum number of concurrent requests.
      num_tests: Number of test images to use.
      vocab_processor : Vocabulary processor for chat inputs
    Returns:
      The classification error rate.

    Raises:
      IOError: An error occurred processing test data set.
    """
    test_data_set = ['Madar.']
    test_data = numpy.array(list(vocab_processor.fit_transform(test_data_set)))
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)
    print("running for ", num_tests)
    start_time = dt.now()
    for _ in range(num_tests):
        sleep(0.01)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'abuse'
        request.model_spec.signature_name = 'predict_text'
        request.inputs['text'].CopyFrom(
            tf.contrib.util.make_tensor_proto(test_data[0].astype(numpy.int32), shape=[1, test_data[0].size]))
        result_counter.throttle()
        print((dt.now() - start_time).microseconds//1000)
        start_time = dt.now()
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds

        result_future.add_done_callback(
            _create_rpc_callback(0, result_counter, start_time))
        #print("Adding callback....")
    return result_counter.get_response_time()


def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    start = dt.now()
    do_inference(FLAGS.server, FLAGS.work_dir,
                              FLAGS.concurrency, FLAGS.num_tests, vocab_processor)
    #print('\nInference error rate: %s%%' % (error_rate * 100))
    end = dt.now()
    print("Time taken: ", (end-start).seconds)

if __name__ == '__main__':
    tf.app.run()
