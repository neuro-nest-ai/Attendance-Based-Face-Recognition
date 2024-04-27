import cv2
import os

# Function to detect faces and save them
def detect_and_save_faces():
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Create directory to save faces
    os.makedirs("faces", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Prompt user to enter name of the person
            name = input("Enter the name of the person (or 'q' to quit): ")
            if name.lower() == 'q':
                cap.release()
                cv2.destroyAllWindows()
                return

            # Create directory for the person's faces
            person_dir = os.path.join("faces", name)
            os.makedirs(person_dir, exist_ok=True)

            # Save the face
            face_filename = os.path.join(person_dir, f"face_{len(os.listdir(person_dir))}.jpg")
            cv2.imwrite(face_filename, gray[y:y+h, x:x+w])

        # Display the frame
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_save_faces()
