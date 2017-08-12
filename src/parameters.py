
class Prms:
    '''
    Global definition of the hog parameters 
    to be used consistently throughout the project
    '''
    
    # Running mode
    DEBUG           = False
    
    # Hog parameters
    COLORSPACE      = 'LUV'
    ORIENT          = 8
    PIX_PER_CELL    = 8
    CELL_PER_BLOCK  = 2
    HOG_CHANNEL     = 0
    SPATIAL_SIZE    = (16, 16)
    N_BINS          = 32

    # Feature flags
    SPATIAL_FEAT    = True
    HIST_FEAT       = True
    HOG_FEAT        = True
    
    # Hog subsampling
    IMAGE_THRESHOLD = 3
    VIDEO_THRESHOLD = 25
    
    # Sliding window
    XY_WINDOW       = (128, 128)
    XY_OVERLAP      = (0.85, 0.85)
    
    # Lists and their indices
    Y_START         = [380, 400, 500]
    Y_STOP          = [480, 600, 700]
    SCALE           = [1.0, 1.5, 2.5]
    FAR             = 0
    MID             = 1
    NEAR            = 2

    # Heatmap
    FRAMES_MAX      = 5
