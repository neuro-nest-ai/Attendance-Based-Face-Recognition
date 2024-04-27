import os
import cv2
import numpy as np
import face_recognition
import threading
import csv
from datetime import datetime

# Function to capture and process frames
def process_frames(cap, detection_zone, known_face_encodings, known_face_names):
    # Create or append to the attendance log file
    with open('attendance_log.csv', mode='a+', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is empty
        if os.stat('attendance_log.csv').st_size == 0:
            writer.writerow(['Name', 'Time', 'Status'])
    
    while True:
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                   
                    if left >= detection_zone[0] and right <= detection_zone[1] and \
                       top >= detection_zone[2] and bottom <= detection_zone[3]:
                        print(f"{name} entered the detection zone.")
                       
                        log_attendance(name)
                        
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
          
            cv2.rectangle(frame, (detection_zone[0], detection_zone[2]), 
                          (detection_zone[1], detection_zone[3]), (0, 255, 0), 2)
            
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Failed to capture frame from the camera stream.")
            break

# Function to log attendance in CSV file
def log_attendance(name):
    with open('attendance_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Login"])

# Prepare images and names of persons to recognize
known_face_encodings = []
known_face_names = []
main_dir = r"D:\NeuroNestAI\Projects2024\Attendance Based Face Recognition\Images"
for root, dirs, files in os.walk(main_dir):
    for name in dirs:
        folder_path = os.path.join(root, name)
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in images:
            image_path = os.path.join(folder_path, image_file)
            person_name = os.path.splitext(image_file)[0]
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)

# Initialize RTSP stream URL
rtsp_url = "rtsp://admin:FPEWBO@192.168.0.101:554/1"
cap = cv2.VideoCapture(rtsp_url)

# Define the coordinates of the detection zone (left, right, top, bottom)
detection_zone = (1300, 1550, 650, 150)

# Start a separate thread for processing frames
frame_thread = threading.Thread(target=process_frames, args=(cap, detection_zone, known_face_encodings, known_face_names))
frame_thread.start()

# Release the capture and close OpenCV windows
frame_thread.join()
cap.release()
cv2.destroyAllWindows()
