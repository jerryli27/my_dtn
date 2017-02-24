import vgg_face
import tensorflow as tf

VGG_FACE_PATH = 'vgg-face.mat'
if __name__ == '__main__':
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        input_images = tf.placeholder(dtype=tf.float32, shape=(1,224,224,3))
        network, average_image, class_names = vgg_face.vgg_face_trainable(VGG_FACE_PATH,input_images)
        print(average_image)
        print(class_names)