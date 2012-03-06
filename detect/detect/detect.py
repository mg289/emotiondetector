from SimpleCV.Display import Display, pg
from SimpleCV import Camera
from faceimage import FaceImage
from const import RES_WIDTH, RES_HEIGHT
import time
import sys

def main():
    """Finds and interprets feature points"""

    # Initialize Camera
    print "Starting Webcam..."
    try:
        cam = Camera()
    except:
	print "Unable to initialize camera"
	sys.exit(1)

    display = Display(resolution = (RES_WIDTH, RES_HEIGHT))
    while not display.isDone():
        # capture the current frame
        try: 
	    capture = cam.getImage()
	    img = capture.smooth()
	except cv.error:
	    print "Camera unsupported by OpenCV"
	    sys.exit(1)

        # Build face and interpret expression
	face_image = FaceImage(img)
	if face_image.face:
	  #s  face_image.interpret()
	    pass

	capture.save(display)
	time.sleep(0.01)
	if display.mouseLeft:
	    display.done = True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
	pass



