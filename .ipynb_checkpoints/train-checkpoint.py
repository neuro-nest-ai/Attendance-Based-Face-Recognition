import cv2
import os

# Function to collect employee face data for training
def collect_face_data(employee_ids, num_samples=50):
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Create directories to store face data
    data_dirs = [f"face_data/{employee_id}" for employee_id in employee_ids]
    for data_dir in data_dirs:
        os.makedirs(data_dir, exist_ok=True)
    
    # Collect face samples for each employee
    for employee_id in employee_ids:
        print(f"Collecting face data for Employee ID {employee_id}...")
        sample_count = 0
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f"face_data/{employee_id}/{sample_count}.jpg", face_img)
                sample_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.waitKey(100)

            cv2.imshow('Collecting Face Data', frame)

            if sample_count >= num_samples:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Face data collection completed.")

if __name__ == "__main__":
    # Collect face data for training
    employee_ids = input("Enter Employee IDs separated by space: ").split()
    collect_face_data(employee_ids)
