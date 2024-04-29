import os
import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Function to log attendance in a CSV file
def log_attendance(name):
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, current_date, current_time])

# Function to update the GUI display
def update_gui(name):
    attendance_label.config(text=f"{name} Present")

# Function to update the camera feed in the GUI
def update_camera_feed(known_face_encodings, known_face_names):
    global attendance_logged
    _, frame = video_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width, height))

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches and not attendance_logged:
            name = known_face_names[matches.index(True)]
            update_gui(name)

            # Update attendance record
            log_attendance(name)
            attendance_logged = True  # Set attendance_logged to True to prevent multiple logging

            # Draw a green rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Draw a red rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    # Convert the frame to PIL format
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)
    if not stop_camera:
        camera_label.after(10, update_camera_feed, known_face_encodings, known_face_names)  # Update every 10 milliseconds

# Function to stop the camera and close the window
def stop_camera():
    global stop_camera
    stop_camera = True
    video_capture.release()
    root.destroy()

# Initialize Tkinter
root = tk.Tk()
root.title("Attendance System")

# Title label
title_label = tk.Label(root, text="Face Recognition Attendance System", font=('Helvetica', 20, 'bold'))
title_label.pack(pady=10)

# Date and time label
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
datetime_label = tk.Label(root, text=current_datetime, font=('Helvetica', 14))
datetime_label.pack()

# Create a label for attendance display
attendance_label = tk.Label(root, text="", font=('Helvetica', 18))
attendance_label.pack(pady=20)

# Create a label for camera feed
camera_label = tk.Label(root)
camera_label.pack()

# Button to stop the camera and close the window
stop_button = tk.Button(root, text="Stop and Exit", command=stop_camera, font=('Helvetica', 14))
stop_button.pack(pady=10)

# Initialize video capture
rtsp_url="rtsp://admin:FPEWBO@192.168.0.100:554/video.cgi?resolution=640x480&req_fps=30&.mjpg"
video_capture = cv2.VideoCapture(rtsp_url)

# Get the initial frame dimensions
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load known face encodings and names
known_face_encodings = []
known_face_names = []
main_dir = r"D:\NeuroNestAI\Projects2024\Attendance Based Face Recognition\Images"
for root_dir, dirs, files in os.walk(main_dir):
    for name in dirs:
        folder_path = os.path.join(root_dir, name)
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in images:
            image_path = os.path.join(folder_path, image_file)
            person_name = os.path.splitext(image_file)[0]
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)

# Flag to stop camera
stop_camera = False
# Flag to check if attendance is logged
attendance_logged = False

# Main loop
update_camera_feed(known_face_encodings, known_face_names)
root.mainloop()
