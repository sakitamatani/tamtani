import os
import math

from keras.callbacks import TensorBoard, ModelCheckpoint

from models import UNet
from utils import SegmentationDataGenerator


def print_log(*log_str):
    print('<train> Log : ', *log_str)

DATA_DIR = 'jikkendata/jikken1/ryousei/data'
TRAIN_BASE_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_INPUTS_DIR = os.path.join(TRAIN_BASE_DIR, 'inputs')
TRAIN_TEACHERS_DIR = os.path.join(TRAIN_BASE_DIR, 'teachers')
VALID_BASE_DIR = os.path.join(DATA_DIR, 'valid')
VALID_INPUTS_DIR = os.path.join(VALID_BASE_DIR, 'inputs')
VALID_TEACHER_DIR = os.path.join(VALID_BASE_DIR, 'teachers')
LOGS_DIR = os.path.join('logs', 'unet')
MODEL_NAME = 'unet_100epoch_lr0.01'
MODEL_FILENAME = os.path.join('results', MODEL_NAME + '.h5')
MODEL_FILENAME_BEST = os.path.join('results', MODEL_NAME + '_best.h5')

ID_CANCER = 1
TARGET_CLASS_IDS = [ID_CANCER]

GPU_NUM = 0
BATCH_SIZE = 64
EPOCHS = 5

model = UNet(len(TARGET_CLASS_IDS) + 1, gpu_num=GPU_NUM)
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

print_log('train_generator.data_size() : ', train_generator.data_size())
print_log('valid_generator.data_size() : ', valid_generator.data_size())

tensorboard_callback = TensorBoard(log_dir=LOGS_DIR)
model_save_callback = ModelCheckpoint(filepath=MODEL_FILENAME_BEST,
                                     verbose=1,
                                     save_best_only=True,
                                     period=5
                                    )

model.fit_generator(
    train_generator.generator(batch_size=BATCH_SIZE),
    steps_per_epoch=math.ceil(train_generator.data_size() / BATCH_SIZE),
    validation_data=valid_generator.generator(batch_size=BATCH_SIZE),
    validation_steps=math.ceil(valid_generator.data_size() / BATCH_SIZE),
    epochs=EPOCHS,
    use_multiprocessing=True,
    callbacks=[tensorboard_callback, model_save_callback]
)

model.save(MODEL_FILENAME)
