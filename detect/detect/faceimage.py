from SimpleCV import * 
from features import Feature, HaarFeature, EyePair, Mouth, Eyebrows
import const

# Load cascade classifiers
face_classifier = cv.Load(const.FACE_PATH)
eye_pair_classifier = cv.Load(const.EYE_PAIR_PATH)
mouth_classifier = cv.Load(const.MOUTH_PATH)

class Emotion:
    def __init__(self, inner_brow_raise = 0, outer_brow_raise = 0, 
		 brow_lower = 0, upper_lip_raise = 0, lip_corner_pull = 0,
		 lip_corner_depress = 0, lip_stretch = 0, lip_press = 0,
		 jaw_drop = 0, dominating = 0):
	# More action units can be added to improve accuracy
	# v_* => vertical distance
	# h_* => horizontal distance
	v_ibr = inner_brow_raise
	v_obr = outer_brow_raise
	v_bl = brow_lower
	v_ulr = upper_lip_raise
	v_lcp = lip_corner_pull
	v_lcd = lip_corner_depress
	h_ls = lip_stretch
	h_lp = lip_press
	v_jd = jaw_drop
	dom = dominating

class FaceImage:
    def __init__(self, img):
	self.image = img.scale(const.IMG_WIDTH, const.IMG_HEIGHT)
	faces = self.image.findHaarFeatures(face_classifier)
	self.face = None
	if faces: 
	    self.face = faces[0]
	else:
	    return

	# commonly used estimates for mouth/eye region
	eye_pair_offset, mouth_offset = 5.5, 1.5 

	# initialize features, localize featurepoints

	self.eye_pair = EyePair(
			self.image, 
			self.face, 
			eye_pair_offset, 
			eye_pair_classifier) 

	self.mouth = Mouth(
			self.image, 
			self.face, 
			mouth_offset, 
			mouth_classifier)
	
	self.eyebrows = Eyebrows(
			   self.image,
			   self.face,
			   self.eye_pair) 
	
	self.emotion = Emotion()

    def interpret(self):
	"""Interprets a faceimage using FACS"""
	eye_pair = self.eye_pair
	mouth = self.mouth
	eyebrows = self.eyebrows

	has_eye_pair = True
	has_mouth = True
        if not eye_pair.cv_feature:
	    has_eye_pair = False
	if not mouth.cv_feature:
	    has_mouth = False

	# calculations from following paper (based on FACS action units): 
	# "Visual-Based Emotion Detection for Natural Man-Machine Interaction"

	neutral = Emotion(const.N_IBR, const.N_OBR, const.N_BL, const.N_ULR, 
			  const.N_LCP, const.N_LCD, const.N_LS, const.N_LP, 
			  const.N_JD)

	normalizer = eye_pair.right_pupil - eye_pair.left_pupil
	eye_x = eye_pair.feature_region.x
	eye_y = eye_pair.feature_region.y 
		
	# See Emotion class for abbreviations
	self.emotion.v_ibr = (eye_y - eyebrows.left_inner_eyebrow[1]) + \
			     (eye_y - eyebrows.right_inner_eyebrow[1])

	self.emotion.v_obr = (eye_y - eyebrows.left_outer_eyebrow[1]) + \
			     (eye_y - eyebrows.right_outer_eyebrow[1])

	# same as brow raise, but becomes active when closer to eye line
	self.emotion.v_bl = self.emotion.v_ibr 

	self.emotion.v_ulr = (mouth.upper_lip[1] - eye_y)

	self.emotion.v_lcp = (mouth.left_corner[1] - eye_y) + \
			     (mouth.right_corner[1] - eye_y)

	self.emotion.v_lcd = self.emotion.v_lcd

	self.emotion.h_ls = (eye_x - mouth.left_corner[0]) + \
			    (mouth.right_corner[0] - eye_x)

	self.emotion.v_jd = (mouth.lower_lip[1] - eye_y)

	ibr = self.emotion.v_ibr
	obr = self.emotion.v_obr
	bl = self.emotion.v_bl
	ulr = self.emotion.v_ulr
	lcp = self.emotion.v_lcp
	lcd = self.emotion.v_lcd
	ls = self.emotion.h_ls
	jd = self.emotion.v_jd

	# calculate level of activity for each emotion 
	# formulas based on FACS action units	
	emotion_activities = [
	    #fear
	    (ibr + obr + ls)/6,

	    #surprise
	    (ibr + obr + 2*jd)/6,

	    # anger
	    (bl + lp)/4,

	    # sadness 
	    (bl + lcd)/4,

	    # disgust
	    (ul + ulr)/3,

	    # happiness
	    lcp/2
	]
	
	self.emotion.dom = max(emotion_activities)
