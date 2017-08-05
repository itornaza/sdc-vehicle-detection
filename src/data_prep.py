import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dip import dip


#-----------------------
# Hog hyper-parameters
#-----------------------

COLORSPACE = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 0 # Can be 0, 1, 2, or "ALL"

def _get_data_from_file():
    '''
    Reads the images from the ./dataset directory and 
    returns a list with the car images and a list with then non-car images
    '''
    
    # Lists to hold the car and non car images
    car_images = []
    non_car_images = []
    
    # Get the car images from the dataset
    gti_far_path = '../dataset/vehicles/GTI_Far'
    gti_far_images = glob.glob(os.path.join(gti_far_path, '*.png'))
    gti_left_path = '../dataset/vehicles/GTI_Left'
    gti_left_images = glob.glob(os.path.join(gti_left_path, '*.png'))
    gti_middle_path = '../dataset/vehicles/GTI_MiddleClose'
    gti_middle_images = glob.glob(os.path.join(gti_middle_path, '*.png'))
    gti_right_path = '../dataset/vehicles/GTI_Right'
    gti_right_images = glob.glob(os.path.join(gti_right_path, '*.png'))
    kitti_path = '../dataset/vehicles/KITTI_extracted'
    kitti_images = glob.glob(os.path.join(kitti_path, '*.png'))
    
    # Collect results to the cars list
    car_images = gti_far_images + gti_left_images + gti_middle_images + gti_right_images + kitti_images
    
    # Get the non-car images from the dataset
    extras_path = '../dataset/non-vehicles/Extras'
    extras_images = glob.glob(os.path.join(extras_path, '*.png'))
    gti_path = '../dataset/non-vehicles/GTI'
    gti_images = glob.glob(os.path.join(gti_path, '*.png'))
    
    # Collect results to the non-cars list
    non_car_images = extras_images + gti_images

    # Return a list for the car and a list for the non-car images
    return car_images, non_car_images

def _data_look(car_list, notcar_list):
    '''Returns some characteristics of the dataset'''
        
    data_dict = {}
    
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    
    # Read in a test image, either car or notcar
    example_img = cv2.imread(car_list[0])
    
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    
    # Return data_dict
    return data_dict

def _show_stats(data_info):
    print('>>> Your function returned a count of', data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:', data_info["data_type"])

def _get_random_images(cars, notcars):
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))
    
    # Read in car / not-car images
    car_image = cv2.imread(cars[car_ind])
    notcar_image = cv2.imread(notcars[notcar_ind])

    return car_image, notcar_image, car_ind

def _hog(car_image):
    '''Applies the hog filter to the image'''

    # Convert image to grayscale
    gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
    
    # Define HOG parameters from the globals
    orient = ORIENT
    pix_per_cell = PIX_PER_CELL
    cell_per_block = CELL_PER_BLOCK

    # Hog processing
    features, hog_image = dip.get_hog_features(gray,
                                               orient,
                                               pix_per_cell,
                                               cell_per_block,
                                               vis=True,
                                               feature_vec=False)
   # Return the features and the image
    return features, hog_image

def _get_combined_features(cars, notcars):
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

    # TODO: combine all filters features
    np.append(car_features, car_features_hog)
    np.append(notcar_features, notcar_features_hog)
    
    print('>>> Using:',orient,'orientations',pix_per_cell,
          'pixels per cell and', cell_per_block,'cells per block')
    
    return car_features, notcar_features
    
def _normalize_features(car_features, notcar_features):
    '''Normalize the features'''
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    return X, scaled_X
    
def _processed_dataset(X, scaled_X, car_features_n, notcar_features_n):
    '''Split the dataset into training and test set'''
    
    # Define the labels vector
    y = np.hstack((np.ones(car_features_n), np.zeros(notcar_features_n)))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2,
                                                        random_state=rand_state)

    return X_train, X_test, y_train, y_test

def _plot_car_notcar(car_image, notcar_image):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    plt.show()

def _plot_hog(car_image, hog_image):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()

def _plot_normalized_features(X, scaled_X, car_image, car_ind):
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(car_image)
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.show()

def data_prep(vis=True):
    '''Explore the dataset and return the cars and not cars images in two different lists'''
    
    # Get the car and notcar images from the dataset directories
    cars, notcars = _get_data_from_file()
    
    # Get the color and hog features from the random car image
    car_features, notcar_features = _get_combined_features(cars, notcars)
                                        
    # Normalize the combined features
    X, scaled_X = _normalize_features(car_features, notcar_features)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = _processed_dataset(X, scaled_X,
                                                          len(car_features),
                                                          len(notcar_features))

    # Show results by default
    if vis:
        # Get some information about the dataset size
        data_info = _data_look(cars, notcars)
        
        # Get a random car and a random notcar image for display purposes
        car_image, notcar_image, car_ind = _get_random_images(cars, notcars)
        
        # Get the hog features of the random car image
        hog_image_features, hog_image = _hog(car_image)
        
        # Call all ploting functions
        _show_stats(data_info)
        _plot_car_notcar(car_image, notcar_image)
        _plot_hog(car_image, hog_image)
        _plot_normalized_features(X, scaled_X, car_image, car_ind)
        print('>>> Feature vector length:', len(X_train[0]))

    # Return the training and testing datasets to be used by the classifier
    return X_train, X_test, y_train, y_test
