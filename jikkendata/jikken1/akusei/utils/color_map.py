import colorsys



class ColorMap():
    def __init__(self, n=5):
        self.__color_map = self.__create(n)


    def __create(self, n):
        rgb_colors = []
        for i in range(n):
            hsv = i/n, 0.8, 1.0
            rgb = colorsys.hsv_to_rgb(*hsv)
            rgb = tuple((int(val * 255) for val in rgb))
            rgb_colors.append(rgb)
        return rgb_colors


    def get(self, k):
        k_ = k % len(self.__color_map)
        return self.__color_map[k_]


    def get_list(self):
        return self.__color_map
