import numpy as np
import cv2


class OpenCVHaarFaceDetector:
    def __init__(self,
                 scaleFactor=1.3,
                 minNeighbors=5,
                 model_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):

        self.face_cascade = cv2.CascadeClassifier(model_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces, _, confidence = self.face_cascade.detectMultiScale3(gray, self.scaleFactor,
                                                                   self.minNeighbors, outputRejectLevels=1)
        faces = [[box[0], box[1], box[2], box[3], c] for box, c in zip(faces, confidence)]

        return np.array(faces)


class OpenCVCaffeeFaceDetector:
    def __init__(self):
        self.dnn = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

    def detect_face(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.dnn.setInput(blob)
        detections = self.dnn.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append([startX, startY, endX - startX, endY - startY, confidence])
        return faces
