"""
The cat face data comes from https://sites.google.com/site/catdatacollection/data
"""

import pickle
import math
import os
import random
import cv2
import numpy as np
from bs4 import BeautifulSoup
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
        try:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except:
            pickle.dump(data, f)
        print ('Saved %s..' %path)

def get_category_name(image_path, category_subdir_index = -3):
    """
    Assume the category name is the dir that contains the image
    :param image_path:
    :return:
    """
    # image_dir = os.path.dirname(image_path).strip('/')
    # image_dir.strip('face').strip('/')
    # category_name = image_dir[image_dir.rfind("/")+1:]
    category_name = image_path.split('/')[category_subdir_index]
    return category_name

def load_cat_and_dog(hw, parent_dir = 'cat_and_dog', save_dir='cnd'):
    # train = load_cat_and_dog_face_from_list(hw, parent_dir=parent_dir,is_trainval=True)
    train, test = load_cat_and_dog_face_from_list(hw, parent_dir=parent_dir)

    # for i in range(train['X'].shape[0]):
    #
    #     cv2.imshow('Hint', cv2.cvtColor(train['X'][i].astype(np.uint8), cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(0)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_pickle(train, os.path.join(save_dir,'train.pkl'))
    save_pickle(test, os.path.join(save_dir,'test.pkl'))

def load_cat_and_dog_face_from_list(hw, parent_dir="cat_and_dog", train_split = 0.9):
    # file_path = os.path.join(parent_dir,"annotations/trainval.txt") if is_trainval else os.path.join(parent_dir,"annotations/test.txt")
    # with open(file_path) as trainval_file:
    #     lines = trainval_file.readlines()
    #     file_names = [l.split(" ")[0] for l in lines]

    file_names = []
    for path, subdirs, files in os.walk(os.path.join(parent_dir,"annotations/xmls")):
        for name in files:
            full_file_path = os.path.join(path, name)
            base, ext = os.path.splitext(full_file_path)
            _, file_name = os.path.split(base)
            if ext == ".xml":
                file_names.append(file_name)

    file_names = sorted(file_names)
    file_names = [name for name in file_names if (
        os.path.isfile(os.path.join(parent_dir, "annotations/xmls", name+'.xml'))
        and os.path.isfile(os.path.join(parent_dir, "images", name+'.jpg')))]

    xml = [load_annotation(os.path.join(parent_dir, "annotations/xmls", p+'.xml')) for p in file_names]
    crop_coordinates = [(int(current_xml.annotation.object.bndbox.xmin.contents[0]),
                         int(current_xml.annotation.object.bndbox.ymin.contents[0]),
                         int(current_xml.annotation.object.bndbox.xmax.contents[0]),
                         int(current_xml.annotation.object.bndbox.ymax.contents[0])) for current_xml in xml]
    categories = [current_xml.annotation.filename.contents[0].split("_")[0] for current_xml in xml]
    unique_categories = sorted(list(set(categories)))
    print("Number of unique categories for cat and dog dataset: %d." %(len(unique_categories)))
    unique_categories_dict = {unique_categories[i]:i for i in range(len(unique_categories))}

    labels = np.array([unique_categories_dict[d] for d in categories])

    image_paths = [os.path.join(parent_dir, "images", p+'.jpg') for p in file_names]
    cropped_images = np.array([np.asarray(
        Image.open(path).convert("RGB").crop(crop_coordinates[path_i]).resize((hw,hw), Image.ANTIALIAS),np.uint8)
                      for path_i, path in enumerate(image_paths)]) # Should I use np.float32??

    num_train = int(cropped_images.shape[0] * train_split) # Select 90% of data as training data.
    train_images = cropped_images[:num_train]
    test_images = cropped_images[num_train:]
    train_labels = labels[:num_train]
    test_labels = labels[num_train:]

    return {'X': train_images, 'y': train_labels},  {'X': test_images, 'y': test_labels}


def load_annotation(path):
    """
    Load annotation file for a given image.
    Args:
        img_name (string): string of the image name, relative to
            the image directory.
    Returns:
        BeautifulSoup structure: the annotation labels loaded as a
            BeautifulSoup data structure
    """
    with open(path) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml)

def main():
    # This is for when I manually separated train and test
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

    # This one won't work because profile data is gray scale..
    # cat_front_dirs = get_all_image_paths_in_dir('CatOpen/')
    # cat_front_images = np.array(read_and_resize_images(cat_front_dirs))
    # cat_left_dirs = get_all_image_paths_in_dir('ProfileData/')
    # cat_left_images = np.array(read_and_resize_images(cat_left_dirs))
    # cat_right_images = cat_left_images[:,:,::-1,:] # flip
    #
    # cat_all_images = np.random.shuffle(np.concatenate((cat_front_images,cat_left_images,cat_right_images)))
    # num_train = cat_all_images.shape[0] * 90 / 100  # Select 90% of data as training data.
    # cat_train_images = cat_all_images[:num_train]
    # cat_test_images = cat_all_images[num_train:]
    #
    # train = {'X': cat_train_images}
    # test = {'X': cat_test_images}
    #
    # if not os.path.exists('cat/'):
    #     os.mkdir('cat/')
    #
    # save_pickle(train, 'cat/train.pkl')
    # save_pickle(test, 'cat/test.pkl')

    # The following is for datasets that has all images in subdirectories representing their categories.
    # Human dataset
    # rootdir = '/mnt/data_drive/home/ubuntu/PycharmProjects/facescrub/resized_224/' # '/mnt/data_drive/home/ubuntu/PycharmProjects/facescrub/download/'  # 'facescrub/'
    # save_dir = '/mnt/data_drive/home/ubuntu/datasets/human_224/'
    # hw = 224
    #
    #
    # face_dirs = [d for d in get_all_image_paths_in_dir(rootdir) if d.split('/')[-2] == "face"]
    # face_categories = [get_category_name(d) for d in face_dirs]
    # face_unique_categories = sorted(list(set(face_categories)))
    # # assert len(face_unique_categories) == 530 # Uncomment this for facescrub dataset sanity check.
    # print("Number of unique categories for human face: %d. Number of images %d" %(len(face_unique_categories), len(face_dirs)))
    # assert face_unique_categories > 0 and face_dirs > 0
    # face_unique_categories_dict = {face_unique_categories[i]:i for i in range(len(face_unique_categories))}
    #
    # random_index = list(range(len(face_dirs)))
    # random.shuffle(random_index)
    # num_train = len(face_dirs) * 90 / 100  # Select 90% of data as training data.
    # train_index = random_index[:num_train]
    # test_index = random_index[num_train:]
    #
    # face_train_dirs = [face_dirs[i] for i in train_index]
    # face_train = np.array(read_and_resize_images(face_train_dirs, height=hw, width=hw), dtype=np.uint8)
    # assert face_train.shape[1] == hw and face_train.shape[2] == hw and face_train.shape[3] == 3
    # face_train_label = np.array([face_unique_categories_dict[d] for d in [face_categories[i] for i in train_index]], dtype=np.uint8)
    # train = {'X': face_train, 'y': face_train_label}
    #
    # face_test_dirs = [face_dirs[i] for i in test_index]
    # face_test = np.array(read_and_resize_images(face_test_dirs, height=hw, width=hw), dtype=np.uint8)
    # assert face_test.shape[1] == hw and face_test.shape[2] == hw and face_test.shape[3] == 3
    # face_test_label = np.array([face_unique_categories_dict[d] for d in [face_categories[i] for i in test_index]], dtype=np.uint8)
    # test = {'X': face_test, 'y': face_test_label}
    #
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # save_pickle(train, os.path.join(save_dir,'train.pkl'))
    # save_pickle(test, os.path.join(save_dir,'test.pkl'))


    # # The following is for datasets that has all images in subdirectories representing their categories.
    # # IT also separates male actors from female ones.
    # # Human dataset
    #
    # for s in ['male','female']:
    #     rootdir = '/mnt/tf_drive/home/ubuntu/PycharmProjects/facescrub/download/'  # 'facescrub/'
    #     hw = 128
    #     save_dir = '/mnt/data_drive/home/ubuntu/datasets/human_%d_%s/' %(hw,s)
    #     with open('facescrub_actors_names.txt' if s == 'male' else 'facescrub_actresses_names.txt', 'r') as f:
    #         current_sex_actors = set([name.strip('\n') for name in f.readlines()])
    #
    #
    #     face_dirs = [d for d in get_all_image_paths_in_dir(rootdir) if d.split('/')[-2] == "face" and d.split('/')[-3] in current_sex_actors]
    #     face_categories = [get_category_name(d) for d in face_dirs]
    #     face_unique_categories = sorted(list(set(face_categories)))
    #     assert len(face_unique_categories) == 530 / 2 # Uncomment this for facescrub dataset sanity check.
    #     print("Number of unique categories for human face: %d. Number of images %d" %(len(face_unique_categories), len(face_dirs)))
    #     assert face_unique_categories > 0 and face_dirs > 0
    #     face_unique_categories_dict = {face_unique_categories[i]:i for i in range(len(face_unique_categories))}
    #
    #     random_index = list(range(len(face_dirs)))
    #     random.shuffle(random_index)
    #     num_train = len(face_dirs) * 90 / 100  # Select 90% of data as training data.
    #     train_index = random_index[:num_train]
    #     test_index = random_index[num_train:]
    #
    #     face_train_dirs = [face_dirs[i] for i in train_index]
    #     face_train = np.array(read_and_resize_images(face_train_dirs, height=hw, width=hw), dtype=np.uint8)
    #     assert face_train.shape[1] == hw and face_train.shape[2] == hw and face_train.shape[3] == 3
    #     face_train_label = np.array([face_unique_categories_dict[d] for d in [face_categories[i] for i in train_index]], dtype=np.uint8)
    #     train = {'X': face_train, 'y': face_train_label}
    #
    #     face_test_dirs = [face_dirs[i] for i in test_index]
    #     face_test = np.array(read_and_resize_images(face_test_dirs, height=hw, width=hw), dtype=np.uint8)
    #     assert face_test.shape[1] == hw and face_test.shape[2] == hw and face_test.shape[3] == 3
    #     face_test_label = np.array([face_unique_categories_dict[d] for d in [face_categories[i] for i in test_index]], dtype=np.uint8)
    #     test = {'X': face_test, 'y': face_test_label}
    #
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     save_pickle(train, os.path.join(save_dir,'train.pkl'))
    #     save_pickle(test, os.path.join(save_dir,'test.pkl'))
    #
    # # Anime face
    # # rootdir = '/mnt/data_drive/home/ubuntu/datasets/animeface-character-dataset/thumb/'  # 'facescrub/'
    # # save_dir = '/mnt/data_drive/home/ubuntu/datasets/anime_face_128/'
    # # hw = 128
    # rootdir = '/mnt/data_drive/home/ubuntu/datasets/animeface-character-dataset/thumb/'  # 'facescrub/'
    # # save_dir = '/mnt/data_drive/home/ubuntu/datasets/anime_face_128/'
    # # hw = 128
    # for hw in [224]:
    #
    #     save_dir = '/mnt/data_drive/home/ubuntu/datasets/anime_face_%d/' %hw
    #     face_dirs = [d for d in get_all_image_paths_in_dir(rootdir)]
    #     face_categories = [get_category_name(d,category_subdir_index=-2) for d in face_dirs]
    #     face_unique_categories = sorted(list(set(face_categories)))
    #     print("Number of unique categories for human face: %d. Number of images %d. They are: %s" %(len(face_unique_categories), len(face_dirs), str(face_unique_categories)))
    #     assert face_unique_categories > 0 and face_dirs > 0
    #     face_unique_categories_dict = {face_unique_categories[i]:i for i in range(len(face_unique_categories))}
    #
    #     random_index = list(range(len(face_dirs)))
    #     random.shuffle(random_index)
    #     num_train = len(face_dirs) * 90 / 100  # Select 90% of data as training data.
    #     train_index = random_index[:num_train]
    #     test_index = random_index[num_train:]
    #
    #     face_train_dirs = [face_dirs[i] for i in train_index]
    #     face_train = np.array(read_and_resize_images(face_train_dirs, height=hw, width=hw), dtype=np.uint8)
    #     assert face_train.shape[1] == hw and face_train.shape[2] == hw and face_train.shape[3] == 3
    #     face_train_label = np.array([face_unique_categories_dict[d] for d in [face_categories[i] for i in train_index]], dtype=np.uint8)
    #     train = {'X': face_train, 'y': face_train_label}
    #
    #     face_test_dirs = [face_dirs[i] for i in test_index]
    #     face_test = np.array(read_and_resize_images(face_test_dirs, height=hw, width=hw), dtype=np.uint8)
    #     assert face_test.shape[1] == hw and face_test.shape[2] == hw and face_test.shape[3] == 3
    #     face_test_label = np.array([face_unique_categories_dict[d] for d in [face_categories[i] for i in test_index]], dtype=np.uint8)
    #     test = {'X': face_test, 'y': face_test_label}
    #
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     save_pickle(train, os.path.join(save_dir,'train.pkl'))
    #     save_pickle(test, os.path.join(save_dir,'test.pkl'))
    # #
    # # # load_cat_and_dog(32, save_dir="cnd_32")
    # # # load_cat_and_dog(128, save_dir="cnd_128")


    # # The following is for combining the human and the anime dataset.
    #
    # human_dir = '/mnt/tf_drive/home/ubuntu/PycharmProjects/my_dtn/human/'
    # anime_dir = '/mnt/data_drive/home/ubuntu/datasets/anime_face_32/'
    # save_dir = '/mnt/data_drive/home/ubuntu/datasets/human_and_anime_face_32/'
    # num_human_category = 530
    # for category in ("train", "test"):
    #     with open(human_dir + category + ".pkl", 'rb') as fh:
    #         human = pickle.load(fh)
    #     with open(anime_dir + category + ".pkl", 'rb') as ah:
    #         anime = pickle.load(ah)
    #     images = np.concatenate((human['X'], anime['X']), axis=0)
    #     labels =  np.concatenate((human['y'], anime['y'] + num_human_category), axis=0)
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     save_pickle({'X': images, 'y': labels}, os.path.join(save_dir, category+'.pkl'))

    # The following is for splitting the 128 dataset so that it can be fit into the memory.

    human_dir = '/mnt/data_drive/home/ubuntu/datasets/human_128/'
    save_dir = '/mnt/data_drive/home/ubuntu/datasets/human_128_split_half/'
    num_human_category = 530
    with open(human_dir + 'train' + ".pkl", 'rb') as fh:
        human_train = pickle.load(fh)
    with open(human_dir + 'test' + ".pkl", 'rb') as ah:
        human_test = pickle.load(ah)
    images = np.concatenate((human_train['X'], human_test['X']), axis=0)
    labels =  np.concatenate((human_train['y'], human_test['y'] + num_human_category), axis=0)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    split = 0.9
    new_train_images = images[:int(human_train['X'].shape[0] * split)]
    new_train_labels = labels[:int(human_train['y'].shape[0] * split)]
    new_test_images = images[int(human_train['X'].shape[0] * split):]
    new_test_labels = labels[int(human_train['y'].shape[0] * split):]
    assert new_train_images.shape[0] == new_train_labels.shape[0] and new_test_images.shape[0] == new_test_labels.shape[0]
    print("Number of train images after the split: %d, number of test images: %d" %(new_train_images.shape[0], new_test_images.shape[0]))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_pickle({'X': new_train_images, 'y': new_train_labels}, os.path.join(save_dir, 'train'+'.pkl'))
    save_pickle({'X': new_test_images, 'y': new_test_labels}, os.path.join(save_dir, 'test'+'.pkl'))
    # pass

    
if __name__ == "__main__":
    main()
    