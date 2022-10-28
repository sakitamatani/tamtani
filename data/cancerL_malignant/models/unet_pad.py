from keras.utils import multi_gpu_model
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

from .layers import PaddedConcatenate


def UNet(class_num, gpu_num=1):
    input_layer = Input(shape=(129, 129, 3))

    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv11)
    pool1 = MaxPooling2D()(conv12)

    conv21 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv22 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv21)
    pool2 = MaxPooling2D()(conv22)

    conv31 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)
    conv32 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv31)

    up1 = UpSampling2D()(conv32)
    concat1 = PaddedConcatenate()([up1, conv22])
    conv41 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)
    conv42 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv41)

    up2 = UpSampling2D()(conv42)
    concat2 = PaddedConcatenate()([up2, conv12])
    conv51 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
    conv52 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv51)

    output_layer = Conv2D(class_num, (1, 1), activation='sigmoid', padding='same')(conv52)

    model = Model(inputs=input_layer, outputs=output_layer)

    if gpu_num >= 2:
        model = multi_gpu_model(model, gpus=gpu_num)

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    return model
