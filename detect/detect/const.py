import os

RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'res')
FACE_PATH = os.path.join(RES_DIR, 'haarcascade_frontalface_alt.xml')
EYE_PAIR_PATH = os.path.join(RES_DIR, 'haarcascade_mcs_eyepair_big.xml')
MOUTH_PATH = os.path.join(RES_DIR, 'haarcascade_mcs_mouth.xml')

IMG_WIDTH = 320
IMG_HEIGHT = 240

RES_WIDTH = 800
RES_HEIGHT = 600

# Threshold for 
EMOT_THRESH = 0.26

# Action unit estimates for neutral face
N_IBR = 0.268
N_OBR = 0.284
N_BL = 0.268 
N_ULR = 0.982
N_LCP = 1.056
N_LCD = 1.056
N_LS = 0.396
N_LP = 0.396
N_JD = 1.223
