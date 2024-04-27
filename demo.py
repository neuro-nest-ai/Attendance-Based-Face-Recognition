import cv2

# Define the RTSP URL
#rtsp_url="rtsp://admin:FPEWBO@192.168.0.101:554/video.cgi?resolution=640x480&req_fps=30&.mjpg"
rtsp_url="rtsp://admin:FPEWBO@192.168.0.101:554/video.cgi?resolution=640x480&req_fps=30&.mjpg"

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the camera.")
    exit()

# Loop to read frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Display the frame
    cv2.imshow('CCTV Camera Feed', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
