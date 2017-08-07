
class Prms():
    '''
    Global definition of the hog parameters to be used consistently
    throughout the project
    '''
    
    COLORSPACE = 'LUV'
    ORIENT = 8
    PIX_PER_CELL = 8
    CELL_PER_BLOCK = 2
    HOG_CHANNEL = 0
    SPATIAL_SIZE = (16, 16)
    N_BINS = 32
    SPATIAL_FEAT = True
    HIST_FEAT = True
    HOG_FEAT = True
