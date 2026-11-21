
class Prms:
    '''
    Global definition of the hog parameters 
    to be used consistently throughout the project
    '''
    
    # Running mode
    DEBUG           = False
    SUBCLIP         = [0, 50] # [0, 50]
    
    # Hog subsampling and heatmap
    IMAGE_THRESHOLD = 2
    VIDEO_THRESHOLD = 28
    FRAMES_MAX      = 10
    
    # Indices for the Y and X lists
    FAR             = 0
    MID             = 1
    NEAR            = 2
    
    # Create 3 search areas depending on the longitudal distance from the car
    Y_START         = [400, 400, 500] # [FAR, MID, NEAR]
    Y_STOP          = [500, 600, 650]
    SCALE           = [1.0, 1.5, 2.5]
    
    # Mask the opposing lane in a left-side car world
    X_START         = [330, 160, 0] # [FAR, MID, NEAR]

    # Sliding window
    XY_WINDOW       = (128, 128)
    XY_OVERLAP      = (0.85, 0.85)

    # Look and feel
    LINE_THICKNESS  = 4
    LINE_COLOR      = (0, 255, 0)

    #----------
    # Caution:
    #----------
    # New training required for parameters change bellow

    # Hog parameters
    COLORSPACE      = 'YCrCb'
    ORIENT          = 9
    PIX_PER_CELL    = 8
    CELL_PER_BLOCK  = 2
    HOG_CHANNEL     = 'ALL'
    SPATIAL_SIZE    = (32, 32)
    N_BINS          = 32
    
    # Feature flags
    SPATIAL_FEAT    = True
    HIST_FEAT       = True
    HOG_FEAT        = True
