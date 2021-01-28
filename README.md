# Mask-Wearing-and-Social-Distancing-Detection-Model

## 1. Business Question
This project aims to use AI technology that can detect whether people are wearing mask correctly, incorrectly, or not wearing mask at all, and at the same time, detect if any two people (or two faces) are staying too close to one another so as to violate the social distancing rule. \
This project can be utilized to help monitor the observance of the mask wearing and social distancing rules, and help prevent the spread of COVID-19.

## 2. Data Collection
With Mask: 1917 \
Incorrect Mask: 1995 \
Without Mask: 1919 \
Total Images: 5831

With Mask and Incorrect Mask Images from cabani \
Without Mask Images from balajisrinivas

## 3. Workflow
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/images(ppt).jpg/p1esnk8v5l184u13jg6len9b1fr4-6.jpg" width = "500">
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/images(ppt).jpg/p1esnk8v5l184u13jg6len9b1fr4-15.jpg" width = "500">

## 4. Face Detection Model
For face detection, we used the DNN Face Detector in OpenCV, which is a Caffe model based on the Single Shot-Multibox Detector (SSD) and uses ResNet-10 architecture as its backbone. It is currently one of the best performing face detection models, especially for detecting faces on videos and can achieve a higher frame rates than other models.

## 5. Pre-trained Model for Mask Wearing Classification
As the training size is small (only 5831 images) , we used a pre-trained model for transfer learning.\
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/images(ppt).jpg/p1esnk8v5l184u13jg6len9b1fr4-8.jpg" width = "500">
We used MobileNet V2 as our pre-trained model, as it is a light weight model with relatively few layers (only 53).  It has a smaller model size with fewer complexity cost, and is suitable to devices with low computational power. This is useful for us as we want to detect faces, and classify the wearing of masks in real time with a MacBook.

## 6. Distance Measurement
After detecting faces with OpenCV, we can draw bounding boxes around the faces and get the coordinates of the faces. The coordinates of the bounding boxes are then used to calculate the distance between people using the Euclidean distance in three dimensions.\
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/images(ppt).jpg/p1esnk8v5l184u13jg6len9b1fr4-11.jpg" width = "500">

Before actually computing the distance measurement, it is first necessary to get the focal length of the webcam by calibrating it with an object with known width and known distance from the webcam.\
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/images(ppt).jpg/p1esnk8v5l184u13jg6len9b1fr4-12.jpg" width = "500">

After getting the focal length, we can get the distance from the webcam, which will be the z coordinate. Then, we can also convert the x and y coordinates from pixels to cm.\
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/images(ppt).jpg/p1esnk8v5l184u13jg6len9b1fr4-13.jpg" width = "500">

## 7. Demo
### Mask Only
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/ppt/media/image14.gif" width = "500">

### Distance Measurement Only
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/ppt/media/image23.gif" width = "500">

### Final
<img src="https://github.com/eddylamhw/Mask-Wearing-and-Social-Distancing-Detection-Model/blob/main/ppt/media/image24.gif" width = "500">

# References
Face Detection Models: Which to Use and Why? \
https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
\
Review: MobileNetV2 — Light Weight Model (Image Classification) \
https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c
\
Find distance from camera to object/marker using Python and OpenCV \
https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
cabani/MaskedFace-Net \
https://github.com/cabani/MaskedFace-Net
\
balajisrinivas/Face-Mask-Detection \
https://github.com/balajisrinivas/Face-Mask-Detection
\
KubraTurker/Social_Distancing-CV \
https://github.com/KubraTurker/Social_Distancing-CV


