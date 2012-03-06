from SimpleCV import *
import math
import warnings
import const

class Region:
    def __init__(self, x=0, y=0, width=0, height=0, img=None):
	self.x = x 
	self.y = y
	self.width = width
	self.height = height
	self.img = img

class Feature:
    """
	Abstract class:
	 The following methods are not implemented
	  - _find_ROI
	  - _create_feature_img
	  - _calc_feature_points
    """
    def __init__(self, img, face):
	self.img = img
	self.face = face

	# ROI: Simulate OpenCV's Region of Interest
	self.ROI = Region()
	
	self.feature_region = Region()	

    def _find_ROI(self):
	"""Estimate Region of Interest"""
	raise NotImplementedError

    def _create_feature_img(self):
	"""Crops original image after localizing feature""" 
	raise NotImplementedError

    def _calc_feature_points(self):
	"""Calculate specific points within feature"""
	raise NotImplementedError

    def _change_context(self, outer_region, inner_region):
	"""
	Change coordinates to apply to original image.
	Needed because SimpleCV does not support ROI
	"""

	# x, y are given as centers-> change to left and top
	x_min = outer_region.x - int(outer_region.width/2)
	y_min = outer_region.y - int(outer_region.height/2)

	# feature/inner coords are given wrt a bounding box
	# x/y_min represents left/top of that bounding box
	# x/y_min are given wrt the original image
	# add them to get absolute coords of inner feature wrt original image

	abs_region = Region( x_min + inner_region.x,
			     y_min + inner_region.y,
			     inner_region.width,
			     inner_region.height,
			     inner_region.img)
	return abs_region

    def _find_ROI_img(self):
	"""Get image using ROI properties"""
	
	ROI = self.ROI

	# Move along, nothing to see here...
	# (issue with float/int even with casts. Unsolved for now)
        warnings.filterwarnings("ignore",category=DeprecationWarning)

	centered = True
	self.ROI.img = self.img.crop(
			    	int(ROI.x), 
			    	int(ROI.y), 
			    	int(ROI.width), 
			    	int(ROI.height), 
			    	centered) 

class HaarFeature(Feature):
    """
	Abstract class:
	 The following method is not implemented
	  - _calc_feature_points
    """
    def __init__(self, img, face, offset, cascade):
	Feature.__init__(self, img, face)

	self.offset = offset
	self.cascade = cascade
	self.cv_feature = None
	self._create_feature_img()

    def _calc_feature_points(self):
	"""Calculate specific points within feature"""
	raise NotImplementedError

    def _find_ROI(self):
	"""Estimate Region of Interest"""
	face = self.face

	face_height = face.height()
	self.ROI.height = int(face_height/3)
	self.ROI.width = face.width()
	self.ROI.x = face.x
	self.ROI.y = face.y - self.ROI.height + int(face_height/self.offset)

    def _create_feature_img(self):
	"""Crops original image after localizing feature""" 

	self._find_ROI()
    	Feature._find_ROI_img(self)
	features = self.ROI.img.findHaarFeatures(self.cascade)
	
	if features:
	    # Initialize properties for feature_region
	    self.cv_feature = features[0]
	    self.feature_region.width = self.cv_feature.width()
	    self.feature_region.height = self.cv_feature.height()

	    # Here we have coords relative to ROI
	    # Calculate coords in original context (ie absolute, not relative)
	    self.feature_region = self._change_context(
					 	self.ROI, 
						self.feature_region)

	    # Create the image
	    self.feature_region.img = self.cv_feature.crop()

class EyePair(HaarFeature):
    def __init__(self, img, face, offset, cascade):
	HaarFeature.__init__(self, img, face, offset, cascade)

	if self.cv_feature:
	    self.left_pupil = (0, 0) 
	    self.right_pupil = (0, 0)
	    self._calc_feature_points()

    def _calc_feature_points(self):
	"""Estimates left and right pupil"""
	
	reg = self.feature_region

	# constants are simply estimates
	segment = int(reg.width/16)
	left_x = int(reg.x - 5*segment)
	right_x = int(reg.x + 5*segment)

	self.left_pupil = (left_x, reg.y)
	self.right_pupil = (right_x, reg.y) 

class Mouth(HaarFeature):
    def __init__(self, img, face, offset, cascade):
	HaarFeature.__init__(self, img, face, offset, cascade)

	if self.cv_feature:
	    self.mouth_img = self.feature_region.img
	    self.left_corner = (0,0)
	    self.right_corner = (0,0)
	    self.upper_lip = (0,0)
	    self.lower_lip = (0,0)
	    self._calc_feature_points()

    def _calc_feature_points(self):
	"""Finds corners, upper and lower lips"""
	
	# lips should be white after binarize
	self.mouth_img = self.mouth_img.binarize()

	# find left and right lip corners 
	
	# todo: add more bounds to improve accuracy
	white = (255, 255, 255)
	dist = self.mouth_img.colorDistance(white).getNumpy()[:,:,0]
	white_pixels = np.where(dist == 0)
	if white_pixels:
	    self.left_corner = (white_pixels[0][0], white_pixels[1][0]) 
	    right = len(white_pixels[0]) - 1
	    self.right_corner = (white_pixels[0][right], white_pixels[1][right]) 
	else:
	    # Guess left-most middle, right-most middle
	    pass

    def transform(self):
	"""
	Transformations to improve search
	"""
	# Use color transform to reduce blue component
	# Blue is not as important for lip segmentation
	imgF = self.mouth_img.getNumpy().astype('float')
	trans = (-imgF[:,:,0] + 2*imgF[:,:,1] - 0.5*imgF[:,:,2])/4
	self.mouth_img = Image(trans)

class Eyebrows(Feature):
    def __init__(self, img, face, eye_pair):
	Feature.__init__(self, img, face)
	self.eye_pair = eye_pair
	self.left_inner_eyebrow = (0,0) 
	self.left_outer_eyebrow = (0,0)
	self.right_inner_eyebrow = (0,0)
	self.right_outer_eyebrow = (0,0)
	self._create_feature_img()

    def _find_ROI(self):
	"""Estimate Region of Interest"""

	eye_pair = self.eye_pair.feature_region
	# All constants are simply common estimates
	if self.eye_pair.cv_feature:
	    # estimate region above eye_pair
	    # make sure ROI does not extend above original image
	    face_top_y = self.face.y - math.floor(self.face.height()/2)
	
	    self.ROI.x = eye_pair.x
	    self.ROI.y = math.ceil((eye_pair.y + face_top_y)/2)   		
	    self.ROI.width = min(
			    int(eye_pair.width*1.15),
			    self.face.width())
	    self.ROI.height = eye_pair.y - face_top_y
	else:
	    # Estimate region without knowledge of eye_pair
	    segment = math.floor(self.face.height()/3)

	    self.ROI.x = self.face.x
	    self.ROI.y = self.face.y - segment
	    self.ROI.width = self.face.width()
	    self.ROI.height = segment

    def _create_feature_img(self):
	"""Crops original image after localizing feature""" 
	self._find_ROI()
	
	# for now, eyebrows region == ROI	
	self.feature_region = self.ROI

    def _calc_feature_points(self):
	"""Calculate specific points within feature"""
	pass

