import cv2

def test_camera_indices():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Testing camera index {i}")
            print("Camera index", i, "works.")
            cap.release()
        else:
            print(f"Camera index {i} does not work.")

if __name__ == "__main__":
    test_camera_indices()
