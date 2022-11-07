import os
import glob
import cv2
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.utils import np_utils


class SegmentationDataGenerator():
    def __init__(self, input_dir, teacher_dir, image_shape,
                    target_class_ids=None, num_classes=None):
        self.__input_dir = input_dir
        self.__teacher_dir = teacher_dir
        self.__image_shape = image_shape
        self.__target_class_ids = target_class_ids
        self.__num_classes = num_classes
        if self.__target_class_ids is not None:
            self.__num_classes = len(self.__target_class_ids) + 1
        if self.__num_classes is None:
            self.__print_err('Number of target classes is unknown')
        self.__update_data_names()


    def __update_data_names(self):
        files = glob.glob(os.path.join(self.__teacher_dir, '*', '*.png'))
        files.sort()
        self.__data_names = []
        for file in files:
            name = os.path.basename(file)
            sub_dir = os.path.basename(os.path.dirname(file))
            input_path = os.path.join(self.__input_dir, sub_dir, name)
            if not os.path.exists(input_path):
                continue
            self.__data_names.append(os.path.join(sub_dir, name))


    def data_size(self):
        return len(self.__data_names)


    def generate_data(self):
        if self.__num_classes is None:
            self.__print_err('Number of target classes is unknown')
            return None, None

        input_list = []
        teacher_list = []
        random.shuffle(self.__data_names)

        pbar = tqdm(total=len(self.__data_names), desc="Generate", unit=" data")
        for name in self.__data_names:
            input_img, teacher_img = self.__load_data(name)
            if input_img is None or teacher_img is None:
                continue

            input_list.append(input_img)
            teacher_list.append(teacher_img)
            pbar.update(1)
        pbar.close()

        inputs = np.array(input_list)
        teachers = np.array(teacher_list)

        return inputs, teachers


    def generator(self, batch_size=None):
        if self.__num_classes is None:
            self.__print_err('Number of target classes is unknown')
            return None

        if batch_size is None:
            batch_size = self.data_size()

        input_list = []
        teacher_list = []
        while True:
            random.shuffle(self.__data_names)

            for name in self.__data_names:
                if len(input_list) >= batch_size:
                    input_list = []
                    teacher_list = []

                input_img, teacher_img = self.__load_data(name)
                if input_img is None or teacher_img is None:
                    continue

                input_list.append(input_img)
                teacher_list.append(teacher_img)

                if len(input_list) >= batch_size:
                    inputs = [np.array(input_list)]
                    teachers = [np.array(teacher_list)]

                    yield inputs, teachers


    def __load_data(self, name):
        input_path = os.path.join(self.__input_dir, name)
        input_img = cv2.imread(input_path)
        if input_img is None:
            return None, None
        input_img = cv2.resize(input_img, self.__image_shape)
        input_img = input_img / 255

        teacher_path = os.path.join(self.__teacher_dir, name)
        teacher_img = Image.open(teacher_path)
        teacher_img = teacher_img.resize(self.__image_shape)
        teacher_img = np.array(teacher_img)

        if self.__target_class_ids is not None or self.__target_class_ids != []:
            cond = np.logical_not(np.isin(teacher_img, self.__target_class_ids))
            teacher_img[cond] = 0
            teacher_img = teacher_img.astype(np.uint16)
            teacher_img *= 100
            for k, cls_id in enumerate(self.__target_class_ids):
                teacher_img[teacher_img == cls_id * 100] = k + 1
        teacher_img = np_utils.to_categorical(teacher_img, num_classes=self.__num_classes)

        return input_img, teacher_img


    def __print_err(self, err_str):
        print('<SegmentationDataGenerator> Error : ', err_str)
