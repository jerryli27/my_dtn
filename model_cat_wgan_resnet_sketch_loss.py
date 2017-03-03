import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from inception_resnet_v1 import inference


class DTN(object):
    """Domain Transfer Network
    """
    def __init__(self, mode='train', learning_rate=0.0003, num_classes = 10, hw = 128, alpha=15, beta=15, gamma=15):
        self.mode = mode
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.hw = hw
        self.alpha = alpha
        self.beta=beta
        self.gamma = gamma # For sketch losses.
        assert hw==128
        
    def content_extractor(self, images, reuse=False):
        # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
        
        if images.get_shape()[3] == 1:
            # For mnist dataset, replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)
        if images.get_shape()[2] <64 or images.get_shape()[1] < 64:
            print("WARNING:resnet may not support images with small size.")
        prelogits, _ = inference(images, keep_probability=1.0,reuse=reuse)
        ret = tf.expand_dims(tf.expand_dims(prelogits, axis=1), axis=2)
        return ret
                
    def generator(self, inputs, reuse=False):
        # inputs: (batch, 1, 1, 128)
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,
                                stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                        stride=1, weights_initializer=tf.contrib.layers.xavier_initializer()):
                        net = slim.conv2d_transpose(inputs, 1024, [4, 4], padding='VALID',
                                                    scope='conv_transpose1_1')  # (batch_size, 4, 4, 1024)
                        net = slim.batch_norm(net, scope='bn1_1')
                        # net = slim.conv2d(net, 1024, [3, 3], scope='conv_transpose1_2')   # (batch_size, 4, 4, 512)
                        # net = slim.batch_norm(net, scope='bn1_2')
                        net = slim.conv2d_transpose(net, 512, [3, 3], scope='conv_transpose2_1')  # (batch_size, 8, 8, 512)
                        net = slim.batch_norm(net, scope='bn2')
                        # net = slim.conv2d(net, 256, [3, 3], scope='conv_transpose2_2')   # (batch_size, 4, 4, 512)
                        # net = slim.batch_norm(net, scope='bn2_2')
                        net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose3_1')  # (batch_size, 16, 16, 256)
                        net = slim.batch_norm(net, scope='bn3')
                        # net = slim.conv2d(net, 128, [3, 3], scope='conv_transpose3_2')   # (batch_size, 4, 4, 512)
                        # net = slim.batch_norm(net, scope='bn3_2')
                        net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose4_1')  # (batch_size, 32, 32, 128)
                        net = slim.batch_norm(net, scope='bn4')
                        net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose5_1')  # (batch_size, 64, 64, 128)
                        net = slim.batch_norm(net, scope='bn5')
                        net = slim.conv2d_transpose(net, 3, [3, 3], activation_fn=tf.nn.tanh, scope='conv_transpose4')   # (batch_size, 128, 128, 3)
                        return net
    
    def discriminator(self, images, var_scope = 'discriminator', reuse=False):
        # images: (batch, 32, 32, 3)
        with tf.variable_scope(var_scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                    # net = slim.conv2d(images, 128, [3, 3], stride=1, activation_fn=tf.nn.relu,
                    #                   scope='conv1_1')  # (batch_size, 32, 32, 128)
                    # net = slim.batch_norm(net, scope='bn1_1')
                    # net = slim.conv2d(net, 128, [3, 3], stride=2, scope='conv1_2')  # (batch_size, 16, 16, 128)
                    # net = slim.batch_norm(net, scope='bn1_2')
                    # net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv2_1')  # (batch_size, 16, 16, 256)
                    # net = slim.batch_norm(net, scope='bn2_1')
                    # net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv2_2')  # (batch_size, 8, 8, 256)
                    # net = slim.batch_norm(net, scope='bn2_2')
                    # net = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv3_1')  # (batch_size, 8, 8, 512)
                    # net = slim.batch_norm(net, scope='bn3_1')
                    # net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3_2')  # (batch_size, 4, 4, 512)
                    # net = slim.batch_norm(net, scope='bn3_2')

                    net = slim.conv2d(images, 128, [4, 4], stride=2, activation_fn=tf.nn.relu,
                                      scope='conv1_1')  # (batch_size, 64, 64, 128)
                    net = slim.batch_norm(net, scope='bn1_1')
                    net = slim.conv2d(net, 128, [4, 4], stride=2, scope='conv2_1')  # (batch_size, 32, 32, 128)
                    net = slim.batch_norm(net, scope='bn2_1')
                    net = slim.conv2d(net, 256, [4, 4], stride=2, scope='conv3_1')  # (batch_size, 16, 16, 256)
                    net = slim.batch_norm(net, scope='bn3_1')
                    net = slim.conv2d(net, 512, [4, 4], stride=2, scope='conv4_1')  # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn4_1')
                    net = slim.conv2d(net, 1024, [4, 4], stride=2, scope='conv5_1')  # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn5_1')
                    net = slim.conv2d(net, 1, [4, 4], padding='VALID', scope='conv6')   # (batch_size, 1, 1, 1)
                    net = slim.flatten(net)
                    return net

    def sketch_extractor(self, images):
        # Must feed in images represented in range 0`255.
        image_shape = images.get_shape().as_list()
        if len(image_shape) != 4:
            raise AssertionError("Input image must have shape [batch_size, height, width, 1 or 3]")
        if image_shape[3] == 3:
            gray_images = tf.image.rgb_to_grayscale(images)
        elif image_shape[3] == 1:
            gray_images = images
        else:
            raise AssertionError("Input image must have shape [batch_size, height, width, 1 or 3]")
        filt = np.expand_dims(np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                            np.uint8),axis=2)
        stride = 1
        rate = 1
        padding = 'SAME'
        dil = tf.nn.dilation2d(gray_images, filt, (1,stride,stride,1), (1,rate,rate,1), padding, name='image_dilated')
        sketch = 255 - tf.abs(gray_images - dil)
        # Did NOT apply a threshold here to clear out the low values because i think it may not be necessary.
        assert sketch.get_shape().as_list() == gray_images.get_shape().as_list()
        return sketch
                
    def build_model(self):
        
        if self.mode == 'pretrain':
            self.images = tf.placeholder(tf.float32, [None, self.hw , self.hw , 3], 'svhn_images')
            self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            
            # logits and accuracy
            self.logits = self.content_extractor(self.images)
            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            # loss and train op
            self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
            
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        elif self.mode == 'eval':
            self.images = tf.placeholder(tf.float32, [None, self.hw , self.hw , 3], 'svhn_images')

            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.images)
            self.sampled_images = self.generator(self.fx)

        elif self.mode == 'train':
            self.src_images = tf.placeholder(tf.float32, [None, self.hw , self.hw , 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, self.hw , self.hw , 3], 'mnist_images')

            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.src_images)
            self.fake_images = self.generator(self.fx)
            self.logits = self.discriminator(self.fake_images)
            self.fake_sketches = self.sketch_extractor((self.fake_images + 1) * 127.5)
            self.fake_sketches_logits = self.discriminator(self.fake_sketches, var_scope='discriminator_sketch')
            self.fgfx = self.content_extractor(self.fake_images, reuse=True)

            # loss
            # self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.zeros_like(self.logits))
            # self.g_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.ones_like(self.logits))
            self.d_loss_src = tf.reduce_mean(self.logits) + tf.reduce_mean(self.fake_sketches_logits)
            self.g_loss_src = - tf.reduce_mean(self.logits) - tf.reduce_mean(self.fake_sketches_logits)
            self.f_loss_src = tf.reduce_mean(tf.square(self.fx - self.fgfx)) * self.alpha
            
            # optimizer
            self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            
            t_vars = tf.trainable_variables()
            # 'discriminator' should include vars for sketch discriminator as well.
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            f_vars = [var for var in t_vars if 'InceptionResnetV1' in var.name]

            # TODO: add weight clipping

            self.d_clip_ops = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in d_vars]

            # train op
            with tf.name_scope('source_train_op'):
                self.d_train_op_src = slim.learning.create_train_op(self.d_loss_src, self.d_optimizer_src, variables_to_train=d_vars)
                self.g_train_op_src = slim.learning.create_train_op(self.g_loss_src, self.g_optimizer_src, variables_to_train=g_vars)
                self.f_train_op_src = slim.learning.create_train_op(self.f_loss_src, self.f_optimizer_src, variables_to_train=f_vars)
            
            # summary op
            d_loss_src_summary = tf.summary.scalar('src_d_loss', self.d_loss_src)
            g_loss_src_summary = tf.summary.scalar('src_g_loss', self.g_loss_src)
            f_loss_src_summary = tf.summary.scalar('src_f_loss', self.f_loss_src)
            origin_images_summary = tf.summary.image('src_origin_images', self.src_images)
            sampled_images_summary = tf.summary.image('src_sampled_images', self.fake_images)
            sampled_images_sketches_summary = tf.summary.image('src_sampled_images_sketches', self.fake_sketches)
            self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary, 
                                                    f_loss_src_summary, origin_images_summary, 
                                                    sampled_images_summary, sampled_images_sketches_summary])
            
            # target domain (mnist)
            self.fx = self.content_extractor(self.trg_images, reuse=True)
            self.reconst_images = self.generator(self.fx, reuse=True)
            self.logits_fake = self.discriminator(self.reconst_images, reuse=True)
            self.logits_real = self.discriminator(self.trg_images, reuse=True)
            self.reconst_images_sketches = self.sketch_extractor((self.reconst_images + 1) * 127.5)
            self.reconst_images_sketches_logits = self.discriminator(self.reconst_images_sketches, var_scope='discriminator_sketch', reuse=True)
            self.trg_images_sketches = self.sketch_extractor((self.trg_images + 1) * 127.5)
            self.trg_images_sketches_logits = self.discriminator(self.trg_images_sketches, var_scope='discriminator_sketch', reuse=True)
            
            # loss
            # self.d_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            # self.d_loss_real_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
            self.d_loss_fake_trg = tf.reduce_mean(self.logits_fake)
            self.d_loss_real_trg = - tf.reduce_mean(self.logits_real)
            self.d_loss_fake_trg_sketch = tf.reduce_mean(self.reconst_images_sketches_logits)
            self.d_loss_real_trg_sketch = - tf.reduce_mean(self.trg_images_sketches_logits)
            self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg + self.d_loss_fake_trg_sketch + self.d_loss_real_trg_sketch
            self.g_loss_fake_trg = - tf.reduce_mean(self.logits_fake)
            self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_images - self.reconst_images)) * self.beta
            self.g_loss_const_trg_sketch = tf.reduce_mean(tf.square(self.trg_images_sketches - self.reconst_images_sketches)) * self.gamma
            self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg + self.g_loss_const_trg_sketch
            
            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)

            # train op
            # TODO: add weight clipping
            with tf.name_scope('target_train_op'):
                self.d_train_op_trg = slim.learning.create_train_op(self.d_loss_trg, self.d_optimizer_trg, variables_to_train=d_vars)
                self.g_train_op_trg = slim.learning.create_train_op(self.g_loss_trg, self.g_optimizer_trg, variables_to_train=g_vars)
            
            # summary op
            d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            d_loss_fake_trg_sketch_summary = tf.summary.scalar('trg_d_loss_fake_sketch', self.d_loss_fake_trg_sketch)
            d_loss_real_trg_sketch_summary = tf.summary.scalar('trg_d_loss_real_sketch', self.d_loss_real_trg_sketch)
            d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            g_loss_const_trg_sketch_summary = tf.summary.scalar('trg_g_loss_const_sketch', self.g_loss_const_trg_sketch)
            g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)
            origin_images_summary = tf.summary.image('trg_origin_images', self.trg_images)
            sampled_images_summary = tf.summary.image('trg_reconstructed_images', self.reconst_images)
            origin_images_sketch_summary = tf.summary.image('trg_origin_images_sketch', self.trg_images_sketches)
            sampled_images_sketch_summary = tf.summary.image('trg_reconstructed_images_sketch', self.reconst_images_sketches)
            self.summary_op_trg = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary,
                                                    d_loss_fake_trg_summary, d_loss_real_trg_summary,
                                                    d_loss_fake_trg_sketch_summary, d_loss_real_trg_sketch_summary,
                                                    g_loss_fake_trg_summary, g_loss_const_trg_summary,
                                                    g_loss_const_trg_sketch_summary,
                                                    origin_images_summary, sampled_images_summary,
                                                    origin_images_sketch_summary, sampled_images_sketch_summary])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            