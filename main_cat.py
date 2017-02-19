"""
This repo is taken from https://github.com/yunjey/dtn-tensorflow and modified for research purposes. It is intended for
personal use.
Dataset comes from http://conradsanderson.id.au/lfwcrop/ and http://megaface.cs.washington.edu/participate/challenge.html
"""

import tensorflow as tf
from model_cat import DTN
from solver_cat import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_integer('num_classes', 530, "Number of classes the source dataset have. "
                                         "Should only be changed if not using the default dataset.")
flags.DEFINE_integer('pretrain_iter', 20000, "Number of iteration to run the pretrain mode code.")
flags.DEFINE_integer('train_iter', 5000, "Number of iteration to run the train mode code.")
flags.DEFINE_integer('sample_iter', 100, "Number of iteration to run the sample mode code.")
FLAGS = flags.FLAGS

def main(_):
    
    model = DTN(mode=FLAGS.mode, learning_rate=0.0003, num_classes=FLAGS.num_classes)
    solver = Solver(model, batch_size=100, pretrain_iter=FLAGS.pretrain_iter, train_iter=FLAGS.train_iter, sample_iter=FLAGS.sample_iter,
                    source_dir='human', target_dir='cat', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
    
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