#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from PIL import Image

from experiments.model_cat_wgan_resnet_sketch_loss import *


def _imread(path, shape=None, bw=False, rgba=False, dtype=np.float32):
    # type: (str, tuple, bool, bool) -> np.ndarray
    """

    :param path: path to the image
    :param shape: (Height, width)
    :param bw: Whether the image is black and white.
    :param rgba: Whether the image is in rgba format.
    :return: np array with shape (height, width, num_color(1, 3, or 4))
    """
    assert not (bw and rgba)
    if bw:
        convert_format = 'L'
    elif rgba:
        convert_format = 'RGBA'
    else:
        convert_format = 'RGB'

    if shape is None:
        return np.asarray(Image.open(path).convert(convert_format), dtype)
    else:
        return np.asarray(Image.open(path).convert(convert_format).resize((shape[1], shape[0])), dtype)

class SketchTest(tf.test.TestCase):
    def test_sketch_generation(self):
        with self.test_session() as sess:
            batch_size = 1
            height = None
            width = None
            num_features = 3

            input_layer = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, num_features))
            dtn = DTN()
            input_layer_processed = input_layer / 127.5 - 1
            sketch = dtn.sketch_extractor(input_layer_processed)
            sketch_unprocessed = (sketch + 1) * 127.5


            image_shape = input_layer.get_shape().as_list()
            final_shape = sketch.get_shape().as_list()

            self.assertAllEqual(image_shape[:-1], final_shape[:-1])
            self.assertEqual(1, final_shape[-1])

            sess.run(tf.initialize_all_variables())

            input_image_path =  u'/home/xor/pixiv_images/color/Rella (163536)/14080177_p0 - らくがき.jpg'
            feed_input = np.expand_dims(_imread(input_image_path),axis=0)

            feed_dict = {input_layer:feed_input}
            actual_output = sketch_unprocessed.eval(feed_dict)
            self.assertTrue(actual_output is not None, 'The unet failed to produce an output.')

            cv2.imshow('Input', cv2.cvtColor(feed_input[0], cv2.COLOR_RGB2BGR).astype(np.uint8))
            cv2.imshow('Sketch', actual_output[0].astype(np.uint8))
            cv2.waitKey(0)


if __name__ == '__main__':
    tf.test.main()