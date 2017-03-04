"""
This is taken from https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/examples/detect.py
"""

import cv2
import sys
import os.path
from argparse import ArgumentParser

def get_all_image_paths_in_dir(directory):
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

def detect(directory, save_dir="anime_faces_sanity_check", hw=128, cascade_file="./lbpcascade_animeface.xml"):
    all_img_dirs = get_all_image_paths_in_dir(directory)

    if not os.path.isfile(cascade_file):
        try:
            os.system("wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml")
            assert os.path.isfile(cascade_file)
        except:
            raise RuntimeError("%s: not found" % cascade_file)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    num_faces_detected = 0


    cascade = cv2.CascadeClassifier(cascade_file)
    for filename in all_img_dirs:
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor= 1.1, # 1.1, TODO: try to find the right scale to contain a little bit more features than just the face.
                                         minNeighbors=5,
                                         # minSize=(24, 24))
                                         minSize=(hw / 4, hw / 4))

        for (x, y, w, h) in faces:
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            faces_cropped_image= image[y:y+h, x:x+w]
            faces_cropped_image = cv2.resize(faces_cropped_image, (hw, hw))
            cv2.imwrite(os.path.join(save_dir,"%d.png" %num_faces_detected), faces_cropped_image)
            num_faces_detected += 1

    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #
    # cv2.imshow("AnimeFaceDetect", image)
    # cv2.waitKey(0)
    # cv2.imwrite("out.png", image)


if __name__ == "__main__":
    parser = ArgumentParser()
    # '/home/ubuntu/pixiv_full/pixiv/' or /home/ubuntu/pixiv/pixiv_training_filtered/' or
    # '/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/' -> Number of images  442701.
    parser.add_argument('--directory', dest='directory',
                        help='The path to images containing anime characters. ', metavar='directory',
                        default='/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/')
    parser.add_argument('--save_dir', dest='save_dir',
                        help='The path to save the cropped anime face images. ', metavar='SAVE_DIR',
                        default="anime_faces_sanity_check/",)
    parser.add_argument('--hw', dest='hw', type=int,
                        help='Height and width of images to be saved.', metavar='HW',
                        default=128)
    parser.add_argument('--cascade_file', dest='cascade_file',
                        help='The path to the cascade xml file. ', metavar='CASCADE_FILE',
                        default="./lbpcascade_animeface.xml")
    args = parser.parse_args()

    detect(args.directory, args.save_dir, args.hw, args.cascade_file)