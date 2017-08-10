
from classifier import My_classifier
from data_prep import *
from dip import dip
from parameters import Prms

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

class Pipelines:

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

        for img in glob.glob('../test_images/test*.jpg'):
            image = cv2.imread(img)
            
            # Just run for the middle distance detection
            out_img, box_list = dip.find_cars(image,
                                              Prms.Y_START[Prms.MID],
                                              Prms.Y_STOP[Prms.MID],
                                              Prms.SCALE[Prms.MID],
                                              svc, X_scaler,
                                              Prms.HOG_CHANNEL,
                                              Prms.ORIENT,
                                              Prms.PIX_PER_CELL,
                                              Prms.CELL_PER_BLOCK,
                                              Prms.SPATIAL_SIZE,
                                              Prms.N_BINS)
            
            plt.imshow(out_img)
            plt.show()

    def heat(svc, X_scaler):

        for img in glob.glob('../test_images/test*.jpg'):
            image = cv2.imread(img)

            # Create an empty heat map to draw on
            heat = np.zeros_like(image[:,:,0]).astype(np.float)

            # Get the box list from using the hog sub sampling technique
            out_img, box_list_far = dip.find_cars(image,
                                                  Prms.Y_START[Prms.FAR],
                                                  Prms.Y_STOP[Prms.FAR],
                                                  Prms.SCALE[Prms.FAR],
                                                  svc, X_scaler,
                                                  Prms.HOG_CHANNEL,
                                                  Prms.ORIENT,
                                                  Prms.PIX_PER_CELL,
                                                  Prms.CELL_PER_BLOCK,
                                                  Prms.SPATIAL_SIZE,
                                                  Prms.N_BINS)
            
            out_img, box_list_mid = dip.find_cars(image,
                                                  Prms.Y_START[Prms.MID],
                                                  Prms.Y_STOP[Prms.MID],
                                                  Prms.SCALE[Prms.MID],
                                                  svc, X_scaler,
                                                  Prms.HOG_CHANNEL,
                                                  Prms.ORIENT,
                                                  Prms.PIX_PER_CELL,
                                                  Prms.CELL_PER_BLOCK,
                                                  Prms.SPATIAL_SIZE,
                                                  Prms.N_BINS)
            
            out_img, box_list_near = dip.find_cars(image,
                                                   Prms.Y_START[Prms.NEAR],
                                                   Prms.Y_STOP[Prms.NEAR],
                                                   Prms.SCALE[Prms.NEAR],
                                                   svc, X_scaler,
                                                   Prms.HOG_CHANNEL,
                                                   Prms.ORIENT,
                                                   Prms.PIX_PER_CELL,
                                                   Prms.CELL_PER_BLOCK,
                                                   Prms.SPATIAL_SIZE,
                                                   Prms.N_BINS)

            box_list = box_list_far + box_list_mid + box_list_near

            # Add heat to each box in box list
            heat = dip.add_heat(heat, box_list)

            # Apply threshold to help remove false positives
            heat = dip.apply_threshold(heat, 2)

            # Visualize the heatmap when displaying
            heatmap = np.clip(heat, 0, 255)

            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            draw_img = dip.draw_labeled_bboxes(np.copy(image), labels)

            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            plt.show()
