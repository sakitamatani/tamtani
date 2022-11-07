import math
from keras.layers import ZeroPadding2D, Concatenate


class PaddedConcatenate():
    def __init__(self):
        pass


    def __call__(self, inputs):
        if len(inputs) != 2:
            return inputs

        layer_1, layer_2 = inputs
        h_1, w_1 = layer_1.get_shape().as_list()[1:3]
        h_2, w_2 = layer_2.get_shape().as_list()[1:3]

        pad_hs = self.__get_pad_size(h_1, h_2)
        if h_1 < h_2:
            layer_1 = ZeroPadding2D(padding=(pad_hs, (0, 0)))(layer_1)
        elif h_1 > h_2:
            layer_2 = ZeroPadding2D(padding=(pad_hs, (0, 0)))(layer_2)

        pad_ws = self.__get_pad_size(w_1, w_2)
        if w_1 < w_2:
            layer_1 = ZeroPadding2D(padding=((0, 0), pad_ws))(layer_1)
        elif w_1 > w_2:
            layer_2 = ZeroPadding2D(padding=((0, 0), pad_ws))(layer_2)

        concat = Concatenate()([layer_1, layer_2])
        return concat


    def __get_pad_size(self, target_1, target_2):
        pad = abs((target_2 - target_1) / 2)
        pad_size = (int(pad), int(pad))
        if pad != int(pad):
            pad_size = (math.ceil(pad), int(pad))
        return pad_size
