import os
import numpy as np
from inception_resnet_v1 import *
from download_face_extrator_model import download_and_extract_model

class TestResnet(tf.test.TestCase):
    def test_loading_model(self):
        if not os.path.exists('data/'):
            os.mkdir('data/')
        pretrained_model_name = '20170216-091149' # '20170131-234652'  #
        download_and_extract_model(pretrained_model_name, 'data/')
        model_file = os.path.join('data', pretrained_model_name, 'model-%s.ckpt-250000' % pretrained_model_name)
        hw=128
        with tf.Graph().as_default(), self.test_session() as sess:
            image = tf.placeholder(dtype=tf.float32, shape=(1,hw,hw,3),name='image')
            prelogits, _ = inference(image, keep_probability=1.0)
            # bottleneck = slim.fully_connected(prelogits, args.embedding_size, activation_fn=None,
            #                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            #                                   weights_regularizer=slim.l2_regularizer(args.weight_decay),
            #                                   normalizer_fn=slim.batch_norm,
            #                                   normalizer_params=batch_norm_params,
            #                                   scope='Bottleneck', reuse=False)
            # logits = slim.fully_connected(bottleneck, len(train_set), activation_fn=None,
            #                               weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            #                               weights_regularizer=slim.l2_regularizer(args.weight_decay),
            #                               scope='Logits', reuse=False)
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
            output, = sess.run([prelogits], feed_dict={image:np.zeros((1,hw,hw,3))})
            print(output)
            print(output.shape)

if __name__ == '__main__':
    tf.test.main()
