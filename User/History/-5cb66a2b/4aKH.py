import cv2
import csv

# Initialize list to store coordinates
coordinates = []

# Callback function to get the coordinates on mouse click
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Normalize coordinates by the image size and add to list
        normalized_x = x / frame_width
        normalized_y = y / frame_height
        coordinates.append((normalized_x, normalized_y))
        # Display clicked points on the image
        cv2.circle(frame, (x,y), 5, (0, 255, 0), -1)
        cv2.imshow('image', frame)

        # If 4 points have been selected, write to CSV and exit
        if len(coordinates) == 4:
            with open('roi.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(coordinates)
            print("ROI saved to roi.csv")
            cv2.destroyAllWindows()

# Read the first frame from the video file
cap = cv2.VideoCapture('Sedili.mp4')  # Replace with your video path
ret, frame = cap.read()
if not ret:
    print("Failed to capture video.")
    cap.release()
    exit(0)

frame_height, frame_width = frame.shape[:2]
cap.release()

# Display the image and set mouse callback function
cv2.imshow('image', frame)
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
