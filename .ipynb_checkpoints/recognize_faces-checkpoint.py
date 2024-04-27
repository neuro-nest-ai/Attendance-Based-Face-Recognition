import cv2

# Function to recognize employee face
def recognize_employee(frame):
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # If no faces are detected, return None
    if len(faces) == 0:
        return None
    
    # Assuming only one face is detected, extract the face region
    (x, y, w, h) = faces[0]
    face_img = gray[y:y+h, x:x+w]
    
    # Perform face recognition here using your trained model
    # Replace this section with your face recognition logic
    # For demonstration purposes, let's prompt the user to enter the employee name manually
    employee_name = input("Enter Employee Name: ")
    
    return employee_name

if __name__ == "__main__":
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        employee_name = recognize_employee(frame)
        
        if employee_name is not None:
            print(f"Employee recognized: {employee_name}")
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
