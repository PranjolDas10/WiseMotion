import time
import cv2
import dlib
import numpy as np
from playsound import playsound
import random
from scipy.spatial import distance

# Loading Haar Cascade classifiers for use when analyzing eyes and face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Loading facial landmark predictor by dlib's
predictor = dlib.shape_predictor("/Users/behzadwaseem/DrowsyDetector/shape_predictor_68_face_landmarks.dat")

# Function used to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constant values
EAR_THRESHOLD = 0.25  # if the EAR is below this value, a person can be considered drowsy
CONSEC_FRAMES_DROWSY = 10  # The number of consecutive 'drowsy frames' a person must have in a row before they are declared drowsy
CONSEC_FRAMES_BLINK = 3   # The number of consecutive 'drowsy frames' a person must be below to count the detection as a blink

# list of different warning audios to play
# change to user-specific directory if needed
WARNINGS = ['/Users/behzadwaseem/DrowsyDetector/warning1.wav', '/Users/behzadwaseem/DrowsyDetector/warning2.wav', '/Users/behzadwaseem/DrowsyDetector/warning3.wav']

# Initialize variables to be used when determining drowsiness
frame_counter_drowsy = 0
frame_counter_blink = 0
drowsy = False
blinking = False

# Opening the webcam
cap = cv2.VideoCapture(0)

# Reading webcam footage for drowsiness detection
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Converting footage to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting human faces in greyscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extracting facial features of detected face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye = roi_gray[ey:ey + eh, ex:ex + ew]
            landmarks = predictor(eye, dlib.rectangle(0, 0, ew, eh))
            # Converting detected facial features into numpy array
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

            # Extracting left eye landmarks
            left_eye = landmarks[42:48]
            # Extracting right eye landmarks
            right_eye = landmarks[36:42]

            # Calculating aspect ratios for left and right ear
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Calculating average EAR for both eyes
            avg_ear = (left_ear + right_ear) / 2.0

            # Checking if calculated EAR value is below the drowsiness threshold
            if avg_ear < EAR_THRESHOLD:
                frame_counter_drowsy += 1
                frame_counter_blink += 1
                # Checking for prolonged drowsiness
                if frame_counter_drowsy >= CONSEC_FRAMES_DROWSY:
                    drowsy = True
                    blinking = False
                    cv2.putText(frame, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    warning_audio = str(WARNINGS[random.randint(0,2)])
                    playsound(warning_audio)
                    time.sleep(1)
                # Checking for blinking
                elif frame_counter_blink >= CONSEC_FRAMES_BLINK:
                    blinking = True
                    drowsy = False
            # When not drowsy
            else:
                frame_counter_drowsy = 0
                frame_counter_blink = 0
                drowsy = False
                blinking = False
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Frame for drowsiness detection tab
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
# Ending webcam use
cap.release()
cv2.destroyAllWindows()