from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
import pickle
import face_recognition
import threading
import datetime
from main_app.models import *
from .mailer import *
import time
from django.core.files.base import ContentFile
face_detection_videocam = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
face_detection_webcam = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
prototxtPath = os.path.join(settings.BASE_DIR, "face_detector/deploy.prototxt")
weightsPath = os.path.join(settings.BASE_DIR,'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(os.path.join(settings.BASE_DIR,'face_detector/mask_detector.model'))
data = pickle.loads(open("encodings.pickle", "rb").read())

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        #for (x, y, w, h) in faces_detected:
            #cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        frame_flip = cv2.flip(image,1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()

class IPWebCam(object):
    def __init__(self):
        self.video = cv2.VideoCapture(1)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        #for (x, y, w, h) in faces_detected:
            #cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        frame_flip = cv2.flip(image,1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()

def faceRecog(rgb):
    boxes = face_recognition.face_locations(rgb,model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    name="Unknown"

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],encoding)
                        
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
                    
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

    return name

def getWebcamFrame():
    video=VideoStream(src=0).start()
    time.sleep(3)
    image = video.read()
    video.stop()
    cv2.destroyAllWindows()
    frame = imutils.resize(image, width=650)
    frame = cv2.flip(frame, 1)
    return frame

def getIPWebcamFrame():
    video=VideoStream(src=1).start()
    time.sleep(3)
    image = video.read()
    video.stop()
    cv2.destroyAllWindows()
    frame = imutils.resize(image, width=650)
    frame = cv2.flip(frame, 1)
    return frame

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                    (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    names=[]

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
            if(face.size!=0):
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                pred = maskNet.predict(face)
                name = ""
                if pred[0][0]<pred[0][1]:
                    name = faceRecog(rgb)
                else:
                    name="Unknown"

                faces.append(face)
                locs.append((startX, startY, endX, endY))
                preds.append(pred)
                names.append(name)
    '''
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    '''
    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds,names)

def takeDecisions(frame):
    #cv2.imwrite("C:\\final project\\django\\image.jpg",frame) 
    if frame is None:
        return("Face could not get grabbed")
    (locs, preds , names) = detect_and_predict_mask(frame, faceNet, maskNet)
    for (box, pred , name) in zip(locs, preds,names):
        (startX, startY, endX, endY) = box
        mask=pred[0][0]
        withoutMask=pred[0][1]
        x = datetime.datetime.now()
        print(x,mask,withoutMask,name)
        if mask<withoutMask:
            if name == "Unknown":
                #just enter into DB
                print("name unknown, saving to DB")
                frame = frame[startY:endY, startX:endX]
                ret, buf = cv2.imencode('.jpg', frame)
                content = ContentFile(buf.tobytes())
                v_m=Violators()
                v_m.pic.save('output.jpg',content)
                v_m.save()
                print("Added to DB")
            else:
                #save to DB
                print("Face known - "+name+" Saved to DB")
                frame = frame[startY:endY, startX:endX]
                ret, buf = cv2.imencode('.jpg', frame)
                tbs=cv2.imencode('.jpg', frame)[1].tostring()
                content = ContentFile(buf.tobytes())
                s_m=StudentDetails.objects.get(first_name=name)
                print("This is it ............"+s_m.usn)
                v_m=Violators()
                v_m.usn=s_m.usn
                v_m.pic.save('output.jpg',content)
                v_m.save()
                send_email(s_m.emailid,tbs)
                print("sent mail")