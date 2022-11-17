import cv2
import matplotlib.pyplot as plt

DISPLAY_SIZE = 6


def hide_ax_frame(ax):
        ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        ax.set_xticklabels([])
        plt.sca(ax)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)


class ImageLinePlotter():
    def __init__(self, fig_id=0, plot_area_num=6, display_size=DISPLAY_SIZE):
        self.fig_id = fig_id
        self.display_size = display_size
        self.plot_area_num = plot_area_num
        self.figsize = (self.plot_area_num*self.display_size, self.display_size)
        self.fig = plt.figure(self.fig_id, figsize=self.figsize)
        self.image_infos = [None] * self.plot_area_num
        self.image_count = 0


    def add_image(self, img, title='', pos=None):
        if pos is None:
            pos = len(self.image_infos) - 1
            for k, img_info in enumerate(self.image_infos):
                if img_info is None:
                    pos = k+1
                    break

        self.image_infos[pos-1] = (img, title)


    def show_plot(self):
        for k, img_info in enumerate(self.image_infos):
            if img_info is None:
                continue
            (img, title) = img_info
            ax = self.fig.add_subplot(1, self.plot_area_num, k + 1)
            plt.title(title)
            hide_ax_frame(ax)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
