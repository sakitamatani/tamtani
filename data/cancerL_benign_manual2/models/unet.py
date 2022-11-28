import tensorflow as tf
import keras.backend as KB
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam


def UNet(class_num, gpu_num=1):
    input_layer = Input(shape=(128, 128, 3))

    conv11 = Conv2D(64, (3, 3), padding='same')(input_layer)
    bn11 = BatchNormalization()(conv11)
    act11 = Activation(activation='relu')(bn11)
    conv12 = Conv2D(64, (3, 3), padding='same')(act11)
    bn12 = BatchNormalization()(conv12)
    act12 = Activation(activation='relu')(bn12)
    pool1 = MaxPooling2D()(act12)

    conv21 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn21 = BatchNormalization()(conv21)
    act21 = Activation(activation='relu')(bn21)
    conv22 = Conv2D(128, (3, 3), padding='same')(act21)
    bn22 = BatchNormalization()(conv22)
    act22 = Activation(activation='relu')(bn22)
    pool2 = MaxPooling2D()(act22)

    conv31 = Conv2D(512, (3, 3), padding='same')(pool2)
    bn31 = BatchNormalization()(conv31)
    act31 = Activation(activation='relu')(bn31)
    conv32 = Conv2D(512, (3, 3), padding='same')(act31)
    bn32 = BatchNormalization()(conv32)
    act32 = Activation(activation='relu')(bn32)

    up1 = UpSampling2D()(act32)
    concat1 = Concatenate()([up1, act22])
    conv41 = Conv2D(128, (3, 3), padding='same')(concat1)
    bn41 = BatchNormalization()(conv41)
    act41 = Activation(activation='relu')(bn41)
    conv42 = Conv2D(128, (3, 3), padding='same')(act41)
    bn42 = BatchNormalization()(conv42)
    act42 = Activation(activation='relu')(bn42)

    up2 = UpSampling2D()(act42)
    concat2 = Concatenate()([up2, act12])
    conv51 = Conv2D(64, (3, 3), padding='same')(concat2)
    bn51 = BatchNormalization()(conv51)
    act51 = Activation(activation='relu')(bn51)
    conv52 = Conv2D(64, (3, 3), padding='same')(act51)
    bn52 = BatchNormalization()(conv52)
    act52 = Activation(activation='relu')(bn52)

    output_layer = Conv2D(class_num, (1, 1), activation='sigmoid', padding='same')(act52)

    model = Model(inputs=input_layer, outputs=output_layer)

    if gpu_num >= 2:
        model = multi_gpu_model(model, gpus=gpu_num)

    adam = Adam(lr=0.01)
    dice_loss = DiceLossByClass(input_layer.shape[1:3], class_num).dice_coef_loss
    model.compile(optimizer=adam, loss=dice_loss, metrics=['acc'])

    return model


class DiceLossByClass():
    def __init__(self, input_shape, class_num, ratios=None):
        self.__input_h = input_shape[0]
        self.__input_w = input_shape[1]
        self.__class_num = class_num
        self.__ratios = ratios
        if self.__ratios is None:
            self.__ratios = [3] * len(class_num)


    def dice_coef_loss(self, y_true, y_pred):
        y_trues = self.__separate_by_class(y_true)
        y_preds = self.__separate_by_class(y_pred)

        losses = []
        for y_t, y_p, ratio in zip(y_trues, y_preds, self.__ratios):
            losses.append((1 - self.__dice_coef(y_t, y_p))*ratio)

        return tf.reduce_sum(tf.stack(losses))


    def __separate_by_class(self, y):
        y_res = tf.reshape(y, (-1, self.__input_h, self.__input_w, self.__class_num))
        ys = tf.unstack(y_res, axis=3)
        return ys


    def __dice_coef(self, y_true, y_pred):
        y_true = KB.flatten(y_true)
        y_pred = KB.flatten(y_pred)
        intersection = KB.sum(y_true * y_pred)
        denominator = KB.sum(y_true) + KB.sum(y_pred)
        if denominator == 0:
            return 1
        if intersection == 0:
            return 1 / (denominator + 1)
        return (2.0 * intersection) / denominator
