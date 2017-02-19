"""
The cat face data comes from https://sites.google.com/site/catdatacollection/data
"""

import pickle
import math
import os
import random
import numpy as np
from PIL import Image
from typing import Union, List



def imread(path, shape=None, bw=False, rgba=False, dtype=np.float32):
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
        return np.asarray(Image.open(path).convert(convert_format).resize((shape[1], shape[0]), resample=Image.ANTIALIAS), dtype)


def read_and_resize_images(dirs, height=None, width=None, bw=False, rgba=False):
    # type: (Union[str,List[str]], Union[int,None], Union[int,None], bool, bool) -> Union[np.ndarray,List[np.ndarray]]
    """

    :param dirs: a single string or a list of strings of paths to images.
    :param height: height of outputted images. If height and width are both None, then the image is not resized.
    :param width: width of outputted images. If height and width are both None, then the image is not resized.
    :param bw: Whether the image is black and white
    :param rgba: Whether the image is in rgba format.
    :return: images resized to the specific height or width supplied. It is either a numpy array or a list of numpy
    arrays
    """
    if isinstance(dirs, list):
        images = [read_and_resize_images(d, height, width) for d in dirs]
        return images
    elif isinstance(dirs, str):
        image_1 = imread(dirs)
        # If there is no width and height, we automatically take the first image's width and height and apply to all the
        # other ones.
        if width is not None:
            if height is not None:
                target_shape = (height, width)
            else:
                target_shape = (int(math.floor(float(image_1.shape[0]) /
                                               image_1.shape[1] * width)), width)
        else:
            if height is not None:
                target_shape = (height, int(math.floor(float(image_1.shape[1]) /
                                                       image_1.shape[0] * height)))
            else:
                target_shape = (image_1.shape[0], image_1.shape[1])
        return imread(dirs, shape=target_shape, bw=bw, rgba=rgba)

def get_all_image_paths_in_dir(directory):
    # type: (str) -> List[str]
    """

    :param directory: The parent directory of the images.
    :return: A sorted list of paths to images in the directory as well as all of its subdirectories.
    """
    _allowed_extensions = ['.jpg', '.png', '.JPG', '.PNG']
    if not directory.endswith('/'):
        raise AssertionError('The directory must end with a /')
    content_dirs = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            full_file_path = os.path.join(path, name)
            base, ext = os.path.splitext(full_file_path)
            if ext in _allowed_extensions:
                content_dirs.append(full_file_path)
    if len(content_dirs) == 0:
        raise AssertionError('There is no image in directory %s' % directory)
    content_dirs = sorted(content_dirs)
    return content_dirs

def resize_images(image_arrays, size=[32, 32]):
    # convert float type to integer 
    image_arrays = (image_arrays * 255).astype('uint8')
    
    resized_image_arrays = np.zeros([image_arrays.shape[0]]+size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)
        
        resized_image_arrays[i] = np.asarray(resized_image)
    
    return np.expand_dims(resized_image_arrays, 3)  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def get_category_name(image_path):
    """
    Assume the category name is the dir that contains the image
    :param image_path:
    :return:
    """
    # image_dir = os.path.dirname(image_path).strip('/')
    # image_dir.strip('face').strip('/')
    # category_name = image_dir[image_dir.rfind("/")+1:]
    category_name = image_path.split('/')[-3]
    return category_name

def main():
    # cat_train_dirs = get_all_image_paths_in_dir('CatOpen/train/')
    # cats_train = np.array(read_and_resize_images(cat_train_dirs))
    #
    # train = {'X': cats_train}
    #
    # cat_test_dirs = get_all_image_paths_in_dir('CatOpen/test/')
    # cats_test = np.array(read_and_resize_images(cat_test_dirs))
    # test = {'X': cats_test}
    #
    # if not os.path.exists('cat/'):
    #     os.mkdir('cat/')
    #
    # save_pickle(train, 'cat/train.pkl')
    # save_pickle(test, 'cat/test.pkl')

    face_dirs = [d for d in get_all_image_paths_in_dir('facescrub/') if d.split('/')[-2] == "face"]
    face_categories = [get_category_name(d) for d in face_dirs]
    face_unique_categories = sorted(list(set(face_categories)))
    assert len(face_unique_categories) == 530
    print("Number of unique categories for human face: %d." %(len(face_unique_categories)))
    face_unique_categories_dict = {face_unique_categories[i]:i for i in range(len(face_unique_categories))}

    random_index = list(range(len(face_dirs)))
    random.shuffle(random_index)
    num_train = len(face_dirs) * 90 / 100  # Select 90% of data as training data.
    train_index = random_index[:num_train]
    test_index = random_index[num_train:]

    face_train_dirs = [face_dirs[i] for i in train_index]
    face_train = np.array(read_and_resize_images(face_train_dirs))
    assert face_train.shape[1] == 32 and face_train.shape[2] == 32 and face_train.shape[3] == 3
    face_train_label = np.array([face_unique_categories_dict[d] for d in [face_categories[i] for i in train_index]])
    train = {'X': face_train, 'y': face_train_label}

    face_test_dirs = [face_dirs[i] for i in test_index]
    face_test = np.array(read_and_resize_images(face_test_dirs))
    assert face_test.shape[1] == 32 and face_test.shape[2] == 32 and face_test.shape[3] == 3
    face_test_label = np.array([face_unique_categories_dict[d] for d in [face_categories[i] for i in test_index]])
    test = {'X': face_test, 'y': face_test_label}

    if not os.path.exists('human/'):
        os.mkdir('human/')
    save_pickle(train, 'human/train.pkl')
    save_pickle(test, 'human/test.pkl')

    
if __name__ == "__main__":
    main()
    