from keras.utils import multi_gpu_model
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose


def UNet(class_num, gpu_num=1):
    input_layer = Input(shape=(128, 128, 3))

    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv11)
    pool1 = MaxPooling2D()(conv12)

    conv21 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv22 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv21)
    pool2 = MaxPooling2D()(conv22)

    conv31 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)
    conv32 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv31)

    trans1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu')(conv32)
    concat1 = Concatenate()([trans1, conv22])
    conv42 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)

    trans2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu')(conv42)
    concat2 = Concatenate()([trans2, conv12])
    conv52 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)

    output_layer = Conv2D(class_num, (1, 1), activation='sigmoid', padding='same')(conv52)

    model = Model(inputs=input_layer, outputs=output_layer)

    if gpu_num >= 2:
        model = multi_gpu_model(model, gpus=gpu_num)

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    return model
