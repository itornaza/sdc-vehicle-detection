
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

from parameters import Prms

# Hog hyperparameters
COLORSPACE = Prms.COLORSPACE
ORIENT = Prms.ORIENT
PIX_PER_CELL = Prms.PIX_PER_CELL
CELL_PER_BLOCK = Prms.CELL_PER_BLOCK
HOG_CHANNEL = Prms.HOG_CHANNEL
SPATIAL_SIZE = Prms.SPATIAL_SIZE
N_BINS = Prms.N_BINS

class dip():
    '''Digital Image Processing functions for vehicle detection'''

    def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
        '''
        Takes an image, a list of bounding boxes, and optional color tuple
        and line thickness as inputs then draws boxes in that color on
        the output
        '''

        # Make a copy of the image
        draw_img = np.copy(img)
        
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
        
        # Return the image copy with boxes drawn
        return draw_img

    def find_matches(img, template_list):
        '''
        Searches for template matches and returns a list of bounding boxes
        '''
        
        # Define an empty list to take bbox coords
        bbox_list = []
        
        # Define matching method
        # Alternative options:
        # TM_CCORR_NORMED, TM_CCOEFF, TM_CCORR, TM_SQDIFF, TM_SQDIFF_NORMED
        method = cv2.TM_CCOEFF_NORMED
        
        # Iterate through template list
        for temp in template_list:
            # Read in templates one by one
            tmp = cv2.imread(temp)
            
            # Use cv2.matchTemplate() to search the image
            result = cv2.matchTemplate(img, tmp, method)
            
            # Use cv2.minMaxLoc() to extract the location of the best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Determine a bounding box for the match
            w, h = (tmp.shape[1], tmp.shape[0])
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            # Append bbox position to list
            bbox_list.append((top_left, bottom_right))
        
        # Return the list of bounding boxes
        return bbox_list

    def color_hist(img, nbins=32, bins_range=(0, 256)):
        '''Computes color histogram features'''
        
        # Compute the histogram of the RGB channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def bin_spatial(img, color_space='RGB', size=(32, 32)):
        '''
        Computes color histogram features
        Pass the color_space flag as 3-letter all caps string
        like 'HSV' or 'LUV' etc.
        '''
        
        # Convert image to new color space (if specified)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)
        
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(feature_image, size).ravel()
        
        # Return the feature vector
        return features

    #------------
    # hog
    #------------

    def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        '''Returns hog features and visualization'''
        
        if vis == True:
            features, hog_image = hog(img,
                                      orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      block_norm = 'L2-Hys',
                                      transform_sqrt=True,
                                      visualise=vis,
                                      feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img,
                           orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           block_norm = 'L2-Hys',
                           transform_sqrt=True,
                           visualise=vis,
                           feature_vector=feature_vec)
            return features

    def extract_hog_features(imgs, cspace='RGB', orient=9,
                             pix_per_cell=8, cell_per_block=2, hog_channel=0):
        '''Extracts both color and hog features'''
        
        # Create a list to append feature vectors to
        features = []
        
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = cv2.imread(file)
            
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)
            
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(dip.get_hog_features(feature_image[:,:,channel],
                                                             orient, pix_per_cell, cell_per_block,
                                                             vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = dip.get_hog_features(feature_image[:,:,hog_channel], orient,
                                                    pix_per_cell, cell_per_block, vis=False,
                                                    feature_vec=True)

            # Append the new feature vector to the features list
            features.append(hog_features)

            # Return list of feature vectors
            return features

    def extract_color_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32,
                         hist_range=(0, 256)):
        '''Extract features from a list of images'''
        
        # Create a list to append feature vectors to
        features = []
        
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = cv2.imread(file)
            
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)
            
            # Apply bin_spatial() to get spatial color features
            spatial_features = dip.bin_spatial(feature_image, size=SPATIAL_SIZE)
            
            # Apply color_hist() also with a color space option now
            hist_features = dip.color_hist(feature_image, nbins=N_BINS, bins_range=hist_range)

            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features)))
        
        # Return list of feature vectors
        return features

    def get_combined_features(cars, notcars):
        '''Get the features for the cars and notcars list of images in a combined fashion'''
        
        # Hog parameters set up from globals
        colorspace = COLORSPACE
        orient = ORIENT
        pix_per_cell = PIX_PER_CELL
        cell_per_block = CELL_PER_BLOCK
        hog_channel = HOG_CHANNEL

        # Get the features from the color filter
        car_features = dip.extract_color_features(cars)
        notcar_features = dip.extract_color_features(notcars)
        
        # Get the features from the hog filter
        car_features_hog = dip.extract_hog_features(cars, cspace=colorspace,
                                                    orient=orient,
                                                    pix_per_cell=pix_per_cell,
                                                    cell_per_block=cell_per_block,
                                                    hog_channel=hog_channel)
        notcar_features_hog = dip.extract_hog_features(notcars, cspace=colorspace,
                                                       orient=orient,
                                                       pix_per_cell=pix_per_cell,
                                                       cell_per_block=cell_per_block,
                                                       hog_channel=hog_channel)
        
        np.append(car_features, car_features_hog)
        np.append(notcar_features, notcar_features_hog)
        
        return car_features, notcar_features

    def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
        '''
        Extracts features from a single image window. This function is very 
        similar to extract_features() just for a single image rather than
        list of images
        '''
        
        #1) Define an empty list to receive features
        img_features = []
        
        #2) Apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)
        
        #3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = dip.bin_spatial(feature_image, size=SPATIAL_SIZE)
            
            #4) Append features to list
            img_features.append(spatial_features)
            
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = dip.color_hist(feature_image, nbins=N_BINS)

            #6) Append features to list
            img_features.append(hist_features)

        #7) Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(dip.get_hog_features(feature_image[:,:,channel],
                                                             orient,
                                                             pix_per_cell,
                                                             cell_per_block,
                                                             vis=False,
                                                             feature_vec=True))
            else:
                hog_features = dip.get_hog_features(feature_image[:,:,hog_channel],
                                                    orient,
                                                    pix_per_cell,
                                                    cell_per_block,
                                                    vis=False,
                                                    feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    #---------------------
    # Detecion functions
    #---------------------

    def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        '''
        Takes an image, start and stop positions in both x and y,
        window size (x and y dimensions), and overlap
        fraction (for both x and y)
        '''
        
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

        # Initialize a list to append window positions to
        window_list = []
        
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))

        # Return the list of windows
        return window_list

    def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32),
                   hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
        '''
        Pass an image and the list of windows to be searched (output of slide_windows())
        '''
            
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64),
                                  interpolation=cv2.INTER_AREA)

            #4) Extract features for that window using single_img_features()
            features = dip.single_img_features(test_img, color_space=color_space,
                                               spatial_size=spatial_size, hist_bins=hist_bins,
                                               orient=orient, pix_per_cell=pix_per_cell,
                                               cell_per_block=cell_per_block,
                                               hog_channel=hog_channel,
                                               spatial_feat=spatial_feat,
                                               hist_feat=hist_feat, hog_feat=hog_feat)
            
            
            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            
            #6) Predict using your classifier
            prediction = clf.predict(test_features)
                
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
                    
        #8) Return windows for positive detections
        return on_windows

    def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        '''
        Extracts features using hog sub-sampling and make predictions
        '''
        
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                
                # Get color features
                spatial_features = bin_spatial(subimg, size=SPATIAL_SIZE)
                hist_features = color_hist(subimg, nbins=N_BINS)
                
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

        return draw_img

    #---------
    # Heatmap
    #---------

    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
        # Return updated heatmap
        return heatmap# Iterate through list of bboxes

    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        
        # Return the image
        return img
