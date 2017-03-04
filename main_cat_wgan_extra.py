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
IMDB_WIKI face dataset https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
anime face detector https://github.com/nagadomi/lbpcascade_animeface.git
"""

import tensorflow as tf
from model_cat_wgan_extra import DTN
from solver_cat_wgan_extra import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model_wgan', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample_wgan', "directory for saving the sampled images")
flags.DEFINE_string('source_dir', 'human', "directory to source images saved in two pickel files.")
flags.DEFINE_string('target_dir', 'cat', "directory to target images saved in two pickel files.")
flags.DEFINE_integer('num_classes', 530, "Number of classes the source dataset have. "
                                         "Should only be changed if not using the default dataset.")
flags.DEFINE_integer('pretrain_iter', 20000, "Number of iteration to run the pretrain mode code.")
flags.DEFINE_integer('train_iter', 5000, "Number of iteration to run the train mode code.")
flags.DEFINE_integer('sample_iter', 100, "Number of iteration to run the sample mode code.")
flags.DEFINE_integer('hw', 32, "Height and width of input images.")
flags.DEFINE_integer('batch_size', 100, "batch_size")
flags.DEFINE_float('lr', 0.0003,'Learning rate.')
flags.DEFINE_float('alpha', 15.0,'Learning rate.')
flags.DEFINE_float('beta', 15.0,'Learning rate.')
FLAGS = flags.FLAGS

def main(_):
    model = DTN(mode=FLAGS.mode, learning_rate=FLAGS.lr, num_classes=FLAGS.num_classes, hw=FLAGS.hw, alpha=FLAGS.alpha, beta=FLAGS.beta)
    solver = Solver(model, batch_size=FLAGS.batch_size, pretrain_iter=FLAGS.pretrain_iter, train_iter=FLAGS.train_iter, sample_iter=FLAGS.sample_iter,
                    source_dir=FLAGS.source_dir, target_dir=FLAGS.target_dir, model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)

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

"""
python main_cat_wgan_old.py --mode=train --model_save_path=model_female_to_anime_32_alpha_1 --sample_save_path=sample_female_to_anime_32_alpha_1 --source_dir=human_32_female --target_dir=/mnt/data_drive/home/ubuntu/datasets/anime_face_32 --alpha=1.0
python main_cat_wgan_old.py --mode=train --model_save_path=model_human_to_cat_32_alpha_1 --sample_save_path=sample_human_to_cat_32_alpha_1 --source_dir=human --target_dir=cat --alpha=1.0

python main_cat_wgan_old.py --mode=pretrain --model_save_path=model_pretrain_human_128 --sample_save_path=sample_pretrain_human_128 --source_dir=/mnt/data_drive/home/ubuntu/datasets/human_128 --target_dir=/mnt/data_drive/home/ubuntu/datasets/human_128 --alpha=1.0

"""