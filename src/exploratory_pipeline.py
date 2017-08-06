
import cv2
from dip import dip
import numpy as np
import matplotlib.pyplot as plt

from parameters import Prms

# Hog hyperparameters
COLORSPACE      = Prms.COLORSPACE
ORIENT          = Prms.ORIENT
PIX_PER_CELL    = Prms.PIX_PER_CELL
CELL_PER_BLOCK  = Prms.CELL_PER_BLOCK
HOG_CHANNEL     = Prms.HOG_CHANNEL
SPATIAL_SIZE    = Prms.SPATIAL_SIZE
N_BINS          = Prms.N_BINS
SPATIAL_FEAT    = Prms.SPATIAL_FEAT
HIST_FEAT       = Prms.HIST_FEAT
HOG_FEAT        = Prms.HOG_FEAT


def exploratory_pipeline(svc, X_scaler, vis=False):

    y_start_stop = [400, 640] # Min and max in y to search in slide_window()
    image = cv2.imread('../test_images/test1.jpg')
    draw_image = np.copy(image)

    windows = dip.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))
                           
    hot_windows = dip.search_windows(image, windows, svc, X_scaler,
                                     color_space=COLORSPACE,
                                     spatial_size=SPATIAL_SIZE,
                                     hist_bins=N_BINS,
                                     orient=ORIENT,
                                     pix_per_cell=PIX_PER_CELL,
                                     cell_per_block=CELL_PER_BLOCK,
                                     hog_channel=HOG_CHANNEL,
                                     spatial_feat=SPATIAL_FEAT,
                                     hist_feat=HIST_FEAT,
                                     hog_feat=HOG_FEAT)
                                     
    window_img = dip.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)
    plt.show()
