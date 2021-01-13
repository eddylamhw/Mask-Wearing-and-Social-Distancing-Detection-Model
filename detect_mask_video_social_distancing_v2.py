# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from scipy.spatial import distance
from math import pow, sqrt
import numpy as np
import imutils
import time
import cv2
import os
import time


def detect_and_predict_mask(frame, faceNet, maskNet):
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
	#positions = []
	boxes=[]


	#the following: constants and functions for social distancing
	#calculateConstant_x = 300
	#calculateConstant_y = 615

	#def centroid(startX,startY,endX,endY):
		#centroid_x = round((startX+endX)/2,4)
		#centroid_y = round((startY+endY)/2,4)
		#x = endX-startX
		#y = endY-startY
		#diagonal = x*y/4
		#return centroid_x,centroid_y,diagonal
	
	#def calcDistance(bboxHeight):
		#diagonal = (calculateConstant_x * calculateConstant_y) / bboxHeight
		#return distance

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
			#centroid_x, centroid_y, bboxHeight = centroid(startX,startY,endX,endY)                    
			#distance = calcDistance(bboxHeight)

			# Centroid in centimeter distance
			#centroid_x_centimeters = (centroid_x * distance) / calculateConstant_x
			#centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y


			# add the face and bounding boxes  
			# and social distancing positions
			# to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))
			#positions.append((centroid_x_centimeters, centroid_y_centimeters, distance))

			#new social distancing code
			centerX = (endX + startX)/2
			centerY = (endY + startY)/2
			width = endX-startX
			height = endY-startY

			x = int(centerX - (width/2))
			y = int(centerY - (height/2))

			box_centers = [centerX, centerY]

			boxes.append([x, y, int(width), int(height)])



	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	
	# social distance code
	min_distance = 2000

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

# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt" 
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel" 
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("facemask_eddy_kaggle.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	#frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds, sodists) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred, sodist) in zip(locs, preds, sodists):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(incorrectMask, mask, withoutMask) = pred #(changed code)


		# determine the class label and color we'll use to draw
		# the bounding box and text

		#(the following part is changed to show the 3rd class incorrect Mask)
		if (mask > withoutMask) and (mask >incorrectMask):
			label = "Mask"
		elif (withoutMask>mask) and (withoutMask>incorrectMask):
			label = "No Mask"
		elif (incorrectMask > withoutMask) and (incorrectMask>mask):
			label = "Incorrectly Worn Mask"
		
		if label == "Mask":
			color = (0, 255, 0)
		elif label == "No Mask":
			color = (0, 0, 255)
		else:
			color = (255, 0, 0) #can change the colour for incorrectMask here
		
		if sodist ==1:
			colordist = (0, 0, 255)
			labeldist = "Less than 2m"
		elif sodist ==0:
			colordist = (0, 255, 0)
			labeldist = ""
        
        

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(incorrectMask, mask, withoutMask) * 100) #(changed code)

		# display the label and bounding box rectangle on the output
		# frame

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # display the label and bounding box rectangle on the output (for social distancing)
		# frame
		cv2.putText(frame, labeldist, (startX-20, startY - 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, colordist, 2)
		cv2.rectangle(frame, (startX-20, startY-20), (endX+20, endY+20), colordist, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()