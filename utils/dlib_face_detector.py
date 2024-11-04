import dlib

# Initialize dlib's face detector (HOG-based) and create a facial landmarks predictor
detector = dlib.get_frontal_face_detector()

def detect_faces(image):
    # Detect faces in the grayscale image
    faces = detector(image, 1)
    return faces
