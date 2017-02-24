"""
This repo is taken from https://github.com/yunjey/dtn-tensorflow and modified for research purposes. It is intended for
personal use.
Dataset comes from http://conradsanderson.id.au/lfwcrop/ and http://megaface.cs.washington.edu/participate/challenge.html
animal face http://www.stat.ucla.edu/~zzsi/HiT/exp5.html
cat face https://sites.google.com/site/catdatacollection/data
birds 200 (not face) http://www.vision.caltech.edu/visipedia/CUB-200.html
cat face paper http://mmlab.ie.cuhk.edu.hk/archive/2008/ECCV08_CatDetection.pdf
another cat face database https://www.datainnovation.org/2014/08/10000-cat-pictures-for-science/ (can't open the link)
cat and dog images with face annotation and pixel level annotation http://www.robots.ox.ac.uk/~vgg/data/pets/
anime character face dataset http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/README.html
"""

import tensorflow as tf
from model_cat_wgan_old import DTN
from solver_cat_wgan_old import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_string('target_dir', 'cat', "directory to target images saved in two pickel files.")
flags.DEFINE_integer('num_classes', 530, "Number of classes the source dataset have. "
                                         "Should only be changed if not using the default dataset.")
flags.DEFINE_integer('pretrain_iter', 20000, "Number of iteration to run the pretrain mode code.")
flags.DEFINE_integer('train_iter', 5000, "Number of iteration to run the train mode code.")
flags.DEFINE_integer('sample_iter', 100, "Number of iteration to run the sample mode code.")
flags.DEFINE_integer('hw', 32, "Height and width of input images.")
FLAGS = flags.FLAGS


def main(_):
    model = DTN(mode=FLAGS.mode, learning_rate=0.0003, num_classes=FLAGS.num_classes, hw=FLAGS.hw)
    solver = Solver(model, batch_size=100, pretrain_iter=FLAGS.pretrain_iter, train_iter=FLAGS.train_iter,
                    sample_iter=FLAGS.sample_iter,
                    source_dir='human', target_dir=FLAGS.target_dir, model_save_path=FLAGS.model_save_path,
                    sample_save_path=FLAGS.sample_save_path)

    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.model_save_path):
        tf.gfile.MakeDirs(FLAGS.model_save_path)
    if not tf.gfile.Exists(FLAGS.sample_save_path):
        tf.gfile.MakeDirs(FLAGS.sample_save_path)

    if FLAGS.mode == 'pretrain':
        solver.pretrain()
    elif FLAGS.mode == 'train':
        solver.train()
    else:
        solver.eval()


if __name__ == '__main__':
    tf.app.run()