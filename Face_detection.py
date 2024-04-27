import cv2
import os

# Function to create directory if not exists
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to capture faces from webcam
def capture_faces():
    # Initialize OpenCV's built-in frontal face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract the face region from the frame and resize it
            face = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (100, 100))  # Adjust dimensions as needed

            # Save the face image
            person_name = input("Enter the person's name: ")
            directory = "faces/" + person_name
            create_directory(directory)
            cv2.imwrite(os.path.join(directory, f"{person_name}_{len(os.listdir(directory))}.jpg"), resized_face)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create directories to store faces if not exists
    create_directory("faces")
    # Start capturing faces
    capture_faces()
