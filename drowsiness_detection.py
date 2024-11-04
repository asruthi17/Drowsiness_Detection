import cv2
import dlib
from imutils import face_utils
from utils.eye_aspect_ratio import eye_aspect_ratio
from utils.sound_alarm import sound_alarm
import time
import imutils

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# Initialize the frame counters and the total number of blinks
COUNTER = 0
ALARM_ON = False

# Define the indexes for the facial landmarks corresponding to the left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the video stream and allow the camera sensor to warm up
vs = cv2.VideoCapture(0)
time.sleep(1.0)

try:
    while True:
        # Grab the frame from the video stream, resize it, and convert it to grayscale
        ret, frame = vs.read()
        if not ret:
            print("Failed to capture image")
            break
        
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract the left and right eye coordinates
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]

            # Calculate the eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if the eye aspect ratio is below the blink threshold
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                # If the eyes have been closed for a sufficient number of frames, sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        sound_alarm()

            # Otherwise, reset the counter and alarm state
            else:
                COUNTER = 0
                ALARM_ON = False

            # Visualize the eye aspect ratio
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the video stream and do a bit of cleanup
    vs.release()
    cv2.destroyAllWindows()

