
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from dip import dip
from parameters import Prms

class Pipelines():

    def hot_windows(svc, X_scaler, vis=False):
        '''Check the classifier by applying the vehicle detection to the test images'''

        for img in glob.glob('../test_images/test*.jpg'):
            image = cv2.imread(img)
            draw_image = np.copy(image)

            windows = dip.slide_window(image,
                                       x_start_stop=[None, None],
                                       y_start_stop=[Prms.Y_START, Prms.Y_STOP],
                                       xy_window=Prms.XY_WINDOW,
                                       xy_overlap=Prms.XY_OVERLAP)
                        
            # Keep the windows in each image
            hot_windows = dip.search_windows(image,
                                             windows,
                                             svc,
                                             X_scaler,
                                             color_space=Prms.COLORSPACE,
                                             spatial_size=Prms.SPATIAL_SIZE,
                                             hist_bins=Prms.N_BINS,
                                             orient=Prms.ORIENT,
                                             pix_per_cell=Prms.PIX_PER_CELL,
                                             cell_per_block=Prms.CELL_PER_BLOCK,
                                             hog_channel=Prms.HOG_CHANNEL,
                                             spatial_feat=Prms.SPATIAL_FEAT,
                                             hist_feat=Prms.HIST_FEAT,
                                             hog_feat=Prms.HOG_FEAT)
                    
            # Plot the vehicle detection image
            window_img = dip.draw_boxes(draw_image, hot_windows, color=(255, 0, 0), thick=6)
            plt.imshow(window_img)
            plt.show()

    def hog_sub_sampling(svc, X_scaler):
        '''Apply hog sub-sampling to an image to locate cars with one search'''
        
        img = cv2.imread('../test_images/test1.jpg')
        
        out_img = dip.find_cars(img,
                                Prms.Y_START,
                                Prms.Y_STOP,
                                Prms.SCALE,
                                svc, X_scaler,
                                Prms.HOG_CHANNEL,
                                Prms.ORIENT,
                                Prms.PIX_PER_CELL,
                                Prms.CELL_PER_BLOCK,
                                Prms.SPATIAL_SIZE,
                                Prms.N_BINS)
        
        plt.imshow(out_img)
        plt.show()
