import os
import json
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

dirname = os.path.dirname(__file__)
DATASET = input_data.read_data_sets(os.path.join(dirname, '../data'), one_hot=True)
META = os.path.join(dirname, '../models/mnist.meta')
MODELS = os.path.join(dirname, '../models/')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(META)
    saver.restore(sess, tf.train.latest_checkpoint(MODELS))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    softmax = graph.get_tensor_by_name("softmax:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    feed_dict = { x: DATASET.test.images, y: DATASET.test.labels }

    pred = sess.run([softmax, accuracy], feed_dict=feed_dict)
    with open(os.path.join(dirname, '../metrics/eval.json'), 'w') as outfile:
        json.dump({ "accuracy" : str(pred[1]) }, outfile)

    tf_confusion_matrix = tf.confusion_matrix(labels=tf.argmax(DATASET.test.labels, 1), predictions=tf.argmax(pred[0], 1), num_classes=10)
    tf_confusion_matrix = tf_confusion_matrix.eval()
    
    confusion_matrix = []
    for idx,row in enumerate(tf_confusion_matrix):
        for idy,column in enumerate(row):
            confusion_matrix.append({ "label": "Class " + str(idy), "prediction":  "Class " + str(idx), "count": str(column) })

    with open(os.path.join(dirname, '../metrics/confusion_matrix.json'), 'w') as outfile:
        json.dump(confusion_matrix, outfile)
