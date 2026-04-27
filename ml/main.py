import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
import serial
import time
import urllib.request
import os

model_path = "blaze_face_short_range.tflite"
if not os.path.exists(model_path):
    print("Downloading face detection model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        model_path
    )

ser = serial.Serial('COM6', 9600)
time.sleep(2)

def send_angle(angle):
    angle = int(max(0, min(180, angle)))
    ser.write(f"{angle}\n".encode())

options = vision.FaceDetectorOptions(
    base_options=base_options_module.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE
)

cap = cv2.VideoCapture(1)

with vision.FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.bounding_box
            frame_width = frame.shape[1]

            face_center_x = (bbox.origin_x + bbox.width / 2) / frame_width
            angle = int((1.0 - face_center_x) * 180)
            send_angle(angle)

            cv2.rectangle(frame,
                (bbox.origin_x, bbox.origin_y),
                (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {angle}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Servo Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
ser.close()