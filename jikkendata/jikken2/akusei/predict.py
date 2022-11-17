import os
import cv2
import numpy as np
from keras.models import load_model

from utils import SegmentationDataGenerator, ColorMap


DATA_DIR = 'data'
TRAIN_BASE_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_INPUTS_DIR = os.path.join(TRAIN_BASE_DIR, 'inputs')
TRAIN_TEACHERS_DIR = os.path.join(TRAIN_BASE_DIR, 'teachers')
VALID_BASE_DIR = os.path.join(DATA_DIR, 'valid')
VALID_INPUTS_DIR = os.path.join(VALID_BASE_DIR, 'inputs')
VALID_TEACHER_DIR = os.path.join(VALID_BASE_DIR, 'teachers')
RESULT_BASE_DIR = os.path.join('.', 'results')

MODEL_FILE_NAME = 'unet_100epoch_lr0.01_best.h5'
MODEL_FILE_PATH = os.path.join(RESULT_BASE_DIR, MODEL_FILE_NAME)

ID_CANCER=1

TARGET_CLASS_IDS = [ID_CANCER]


def print_log(*log_str):
    print('<predict> Log : ', *log_str)


def prepare_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def output_results(inputs, preds, fnames, save_dir=RESULT_BASE_DIR, color_map=None):
    for image, pred, fname in zip(inputs, preds, fnames):
        h, w = np.shape(pred)[:2]
        pred_argmax = np.argmax(pred, axis=2)
        if color_map is None:
            color_map = ColorMap(n=len(TARGET_CLASS_IDS)).get_list()
        pred_color = np.zeros((h, w, 3), dtype=np.uint8)
        for i, _ in enumerate(TARGET_CLASS_IDS):
            pred_color[pred_argmax==(i+1)] = color_map[i % len(color_map)]

        save_predict_path = prepare_dir(os.path.join(save_dir, 'predict_outputs'))
        save_overlay_path = prepare_dir(os.path.join(save_dir, 'overlay_outputs'))
        save_argmax_path = prepare_dir(os.path.join(save_dir, 'argmax_outputs'))

        org_img = (image * 255).astype(np.uint8)
        ovry = cv2.addWeighted(org_img, 1, pred_color, 0.6, 0)

        cv2.imwrite(os.path.join(save_predict_path, fname), pred_color)
        cv2.imwrite(os.path.join(save_overlay_path, fname), ovry)
        cv2.imwrite(os.path.join(save_argmax_path, fname), pred_argmax)

    print_log('predicts save to ', save_dir)


model = load_model(MODEL_FILE_PATH)
input_shape = tuple(model.input.shape[1:3])

train_generator = SegmentationDataGenerator(
    TRAIN_INPUTS_DIR,
    TRAIN_TEACHERS_DIR,
    input_shape,
    target_class_ids=TARGET_CLASS_IDS
)

valid_generator = SegmentationDataGenerator(
    VALID_INPUTS_DIR,
    VALID_TEACHER_DIR,
    input_shape,
    target_class_ids=TARGET_CLASS_IDS
)
print_log("###########################################################")
print_log(train_generator.generate_data())
print_log("###########################################################")


t_inputs, t_teachers, t_fnames = train_generator.generate_data()
v_inputs, v_teachers, v_fnames = valid_generator.generate_data()
color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

output_results(t_inputs, model.predict(t_inputs), t_fnames,
                save_dir=os.path.join(RESULT_BASE_DIR, 'train_predicts'),
                color_map=color_map)
output_results(v_inputs, model.predict(v_inputs), v_fnames,
                save_dir=os.path.join(RESULT_BASE_DIR, 'valid_predicts'),
                color_map=color_map)

output_results(t_inputs, t_teachers, t_fnames,
                save_dir=os.path.join(RESULT_BASE_DIR, 'train_teachers'),
                color_map=color_map)
output_results(v_inputs, v_teachers, v_fnames,
                save_dir=os.path.join(RESULT_BASE_DIR, 'valid_teachers'),
                color_map=color_map)
