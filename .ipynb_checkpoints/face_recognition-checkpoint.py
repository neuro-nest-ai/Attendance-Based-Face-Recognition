import cv2
import os
import requests

# Function to recognize faces using API
def recognize_faces_with_api(api_key):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize CCTV camera stream
    cctv_url = "ezopen://open.ezviz.com/[verification code]@[deviceSerial]/[channelNo].hd.live"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        # Get frame from CCTV camera
        response = requests.get(cctv_url, headers=headers)
        frame_bytes = response.content
        frame_array = bytearray(frame_bytes)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # TODO: Implement face recognition to recognize the faces with names
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # TODO: Display name of the recognized person
            
        # Display the frame
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    api_key = "fc0cc6be4b544a1ba22ae1bc85023e12"
    recognize_faces_with_api(api_key)
