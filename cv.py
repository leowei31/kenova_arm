import cv2
import numpy as np

# Define the lower and upper bounds for the white and yellow color ranges in the HSV color space
lower_white = np.array([0, 0, 168])
upper_white = np.array([172, 111, 255])
# lower_yellow = np.array([20, 100, 100])
# upper_yellow = np.array([30, 255, 255])
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([130, 255, 255])


# Create a video capture object for the camera
cap = cv2.VideoCapture(1)

# Continuously read frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV frame to detect white and yellow color ranges
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine the white and yellow binary masks using the bitwise OR operation
    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

    # Find contours in the white binary mask
    contours_white, hierarchy = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours in the yellow binary mask
    contours_yellow, hierarchy = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on shape and area in the white binary mask
    for cnt in contours_white:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
            cv2.putText(frame, 'Target Area', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            print("White Top-left corner: ", top_left)
            print("White Bottom-right corner: ", bottom_right)

    # Filter contours based on shape and area in the yellow binary mask
    for cnt in contours_yellow:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, 'Object', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            top_left = (x, y)
            bottom_right = (x + w, y + h)
            print("Blue Top-left corner: ", top_left)
            print("Blue Bottom-right corner: ", bottom_right)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()