import cv2
import numpy as np

# assign the paths for the modelFile and configFile of the dnn face detection model
modelFile = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"face_detector/deploy.prototxt"

# assign the size that the image is going to be resized to
size=224

#detect the face using the dnn face detection model
def detect_face(modelFile,configFile,size):
    # load the dnn model from the modelFile and configFile
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    # load the image that is used for calculating the focal length
    img = cv2.imread('eddy100(2).jpg')
    # get the shape of the image
    h, w = img.shape[:2]
    # preprocess the image with the blobFromImage function
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (size, size)), 1.0,
    (size, size), (104.0, 177.0, 123.0))
    # input the preprocessed image into the dnn
    net.setInput(blob)
    # run the dnn and obtain the face detections of the image
    faces = net.forward()

    # draw faces on image
    for i in range(faces.shape[2]):
            # extract the confidence (i.e., probability) associated with
		    # the detection
            confidence = faces[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is
		    # greater than 0.5
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
			    # the face
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # draw the bounding boxes around the face detected
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

                # calculate the width of the face in pixels using the x coordinates of the bounding box
                # assuming that there is only one face in the image for calibration
                pixelWidth = endX-startX 
    
    # can uncomment the following to see result of the face detection and the bounding box
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    return pixelWidth

# initialize the known distance from the camera to the face to be 100 cm
knownDistance = 100.0
# initialize the known face width to be 17 cm wide
knownWidth = 17.0

# load the pixel width calculated from the above function, multiplied by the known distance and known width
# and the focal length of the web camera
focalLength = (detect_face(modelFile,configFile,size) * knownDistance) / knownWidth

# print the calculatated focal length of the web camera as the output of this py file
print(focalLength)


