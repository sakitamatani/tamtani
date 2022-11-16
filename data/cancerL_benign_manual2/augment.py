import os
import glob
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def augment(input_dir, teacher_dir, out_inputs_dir, out_teachers_dir,
            color_erase_num=2, gray_erase_num=2):
    files = glob.glob(os.path.join(teacher_dir, '*', '*.png'))
    files.sort()
    pbar = tqdm(total=len(files), desc="Augment", unit=" Files")
    for file in files:
        name = os.path.basename(file)
        sub_dir = os.path.basename(os.path.dirname(file))
        input_path = os.path.join(input_dir, sub_dir, name)
        if not os.path.exists(input_path):
            print('skip : ', input_path)
            pbar.update(1)
            continue
        out_input_dir = os.path.join(out_inputs_dir, sub_dir)
        out_teacher_dir = os.path.join(out_teachers_dir, sub_dir)

        input_img = cv2.imread(input_path)
        teacher_img = np.array(Image.open(file))
        for _ in range(1, color_erase_num):
            imgs, e_infos = random_erase(input_img, teacher_img)
            save_erase_imgs(name, out_input_dir, out_teacher_dir, imgs, e_infos)

        gray_input_img = convert_colorbgr2graybgr(input_img)
        gray_teacher_img = teacher_img
        for _ in range(1, gray_erase_num):
            imgs, e_infos = random_erase(gray_input_img, gray_teacher_img)
            save_erase_imgs(name, out_input_dir, out_teacher_dir, imgs, e_infos)

        pbar.update(1)
    pbar.close()


def random_erase(input_base_img, teacher_base_img,
                erase_pos_min=10, erase_pos_max=400,
                erase_size_min=120, erase_size_max=220):
    epos_h = random.randint(erase_pos_min, erase_pos_max)
    epos_w = random.randint(erase_pos_min, erase_pos_max)
    esize_h = random.randint(erase_size_min, erase_size_max)
    esize_w = random.randint(erase_size_min, erase_size_max)

    erase_teacher = teacher_base_img.copy()
    erase_input = input_base_img.copy()
    erase_teacher[epos_h:epos_h+esize_h, epos_w:epos_w+esize_w] = 0
    erase_input[epos_h:epos_h+esize_h, epos_w:epos_w+esize_w, :] = 0

    return (erase_input, erase_teacher), ((epos_h, epos_w), (esize_h, esize_w))


def save_erase_imgs(name, out_input_dir, out_teacher_dir, imgs, e_infos):
    erase_input, erase_teacher = imgs
    (epos_h, epos_w), _ = e_infos

    fname, ext = os.path.splitext(name)
    aug_fname = fname + '_h%03d_w%03d' % (epos_h, epos_w) + ext

    out_teacher_path = os.path.join(out_teacher_dir, aug_fname)
    if not os.path.exists(out_teacher_dir):
        os.makedirs(out_teacher_dir)

    out_input_path = os.path.join(out_input_dir, aug_fname)
    if not os.path.exists(out_input_dir):
        os.makedirs(out_input_dir)

    Image.fromarray(erase_teacher).save(out_teacher_path)
    cv2.imwrite(out_input_path, erase_input)


def convert_colorbgr2graybgr(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    graybgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return graybgr_img


DATA_DIR = os.path.join('.', 'data')
def SRC_DIR(data_type):
    return os.path.join(DATA_DIR, 'train', data_type)
def DIST_DIR(data_type):
    return os.path.join(DATA_DIR, 'augmented_train', data_type)
INPUTS = 'inputs'
TEACHERS = 'teachers'

augment(SRC_DIR(INPUTS), SRC_DIR(TEACHERS), DIST_DIR(INPUTS), DIST_DIR(TEACHERS))
