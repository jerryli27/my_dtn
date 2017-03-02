import os
import numpy as np
from inception_resnet_v1 import *
from download_face_extrator_model import download_and_extract_model

class TestResnet(tf.test.TestCase):
    def test_loading_model(self):
        pretrained_model_name = '20170216-091149'
        download_and_extract_model(pretrained_model_name, 'data/')
        model_file = os.path.join('data', pretrained_model_name, 'model-%s.ckpt-250000' % pretrained_model_name)
        hw=32
        with tf.Graph() as graph:
            image = tf.placeholder(dtype=tf.float32, shape=(1,hw,hw,3),name='image')
            net, endpoints = inception_resnet_v1(image)
            saver = tf.train.Saver()

            with self.test_session() as sess:
                saver.restore(sess, model_file)
                output = sess.run([net], feed_dict={image:np.zeros((1,hw,hw,3))})
                print(output)
