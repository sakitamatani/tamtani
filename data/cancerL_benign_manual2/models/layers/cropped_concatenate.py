import math
from keras.layers import Cropping2D, Concatenate


class CroppedConcatenate():
    def __init__(self):
        pass


    def __call__(self, inputs):
        if len(inputs) != 2:
            return inputs

        layer_1, layer_2 = inputs
        h_1, w_1 = layer_1.get_shape().as_list()[1:3]
        h_2, w_2 = layer_2.get_shape().as_list()[1:3]

        crop_hs = self.__get_crop_size(h_1, h_2)
        if h_1 < h_2:
            layer_2 = Cropping2D(cropping=(crop_hs, (0, 0)))(layer_2)
        elif h_1 > h_2:
            layer_1 = Cropping2D(cropping=(crop_hs, (0, 0)))(layer_1)

        crop_ws = self.__get_crop_size(w_1, w_2)
        if w_1 < w_2:
            layer_2 = Cropping2D(cropping=((0, 0), crop_ws))(layer_2)
        elif w_1 > w_2:
            layer_1 = Cropping2D(cropping=((0, 0), crop_ws))(layer_1)

        concat = Concatenate()([layer_1, layer_2])
        return concat


    def __get_crop_size(self, target_1, target_2):
        crop = abs((target_2 - target_1) / 2)
        crop_size = (int(crop), int(crop))
        if crop != int(crop):
            crop_size = (math.ceil(crop), int(crop))
        return crop_size
