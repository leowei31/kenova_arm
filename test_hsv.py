import cv2
import numpy as np

def get_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr_pixel = frame[y, x]
        bgr_pixel_2d = np.array([[bgr_pixel]], dtype=np.uint8)
        hsv_pixel = cv2.cvtColor(bgr_pixel_2d, cv2.COLOR_BGR2HSV)
        print("BGR pixel value at (", x, ",", y, "):", bgr_pixel)
        print("HSV pixel value at (", x, ",", y, "):", hsv_pixel)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('frame', frame)

    # Call the function to get pixel value
    cv2.setMouseCallback('frame', get_pixel_value)

    # Press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()


