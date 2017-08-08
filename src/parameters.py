
class Prms():
    '''
    Global definition of the hog parameters 
    to be used consistently throughout the project
    '''
    
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
    
    # Sliding window
    XY_WINDOW       = (128, 128)
    XY_OVERLAP      = (0.85, 0.85)
    
    # Hog subsampling
    Y_START         = 400
    Y_STOP          = 640
    SCALE           = 1.5
