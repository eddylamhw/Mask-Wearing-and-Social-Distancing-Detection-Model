# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 100 cm
KNOWN_DISTANCE = 100.0
# initialize the known object width, which in this case, the piece of
# paper is 10 cm wide
KNOWN_WIDTH = 10.0
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("calibration4.jpg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
print(focalLength)

# loop over the images
for imagePath in sorted(paths.list_images("images")):
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread(imagePath)
	marker = find_marker(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
	# draw a bounding box around the image and display it
	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fft" % (inches / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)



def detect_face(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), #preprocessing function
		(104.0, 177.0, 123.0))

		#The blobFromImage function performs
			# Mean subtraction
			# Scaling
			# normalizing
			# And optionally channel swapping

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	positions = []


	#the following: constants and functions for social distancing
	calculateConstant_x = 20
	calculateConstant_y = 1300

	def centroid(startX,startY,endX,endY):
		centroid_x = round((startX+endX)/2,4)
		centroid_y = round((startY+endY)/2,4)
		bboxHeight = round(endY-startY,4)
		return centroid_x,centroid_y,bboxHeight
	
	def calcDistance(bboxHeight):
		distance = (calculateConstant_x * calculateConstant_y) / bboxHeight
		return distance

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			#getting the social distancing positions
			# Centroid of bounding boxes
			centroid_x, centroid_y, bboxHeight = centroid(startX,startY,endX,endY)                    
			distance = calcDistance(bboxHeight)
			# Centroid in centimeter distance
			centroid_x_centimeters = (centroid_x * distance) / calculateConstant_y
			centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y


			# add the face and bounding boxes  
			# and social distancing positions
			# to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))
			positions.append((centroid_x_centimeters, centroid_y_centimeters, distance))



	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	
	# social distance code
	min_distance = 150

	sodists = [0 for i in range(len(faces))]
	if len(faces) >= 2:
		for i in range(len(faces)-1):
			for j in range(i+1,len(faces)):
				dist = sqrt(pow(positions[i][0]-positions[j][0],2) 
                                          + pow(positions[i][1]-positions[j][1],2) 
                                          + pow(positions[i][2]-positions[j][2],2)
                                          )
				#dist = distance.euclidean(locs[i][:2],locs[j][:2])
				if dist < min_distance:
					sodists[i]=1
					sodists[j]=1

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds, sodists)