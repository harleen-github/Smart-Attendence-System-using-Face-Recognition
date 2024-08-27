import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime, timedelta

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

path = r"C:\Documents\Smart Attendence System\student_images"
studentimg = []
studentNames = []

myList = os.listdir(path)

for cl in myList:
    currImg = cv2.imread(os.path.join(path, cl))
    if currImg is not None:
        studentimg.append(currImg)
        studentNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Error loading image {cl}")

def finEncoding(images):
    imgEncodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_rec.face_encodings(img)
        if encodings:
            encodeimg = encodings[0]
            imgEncodings.append(encodeimg)
        else:
            print("No face found in the image")
    return imgEncodings

attendance_time_limit = timedelta(minutes=1)
last_attendance_time = {}

# Set to keep track of processed names in the current session
processed_names = set()

def MarkAttendance(name):
    if name in processed_names:
        return  # Skip if the name has already been processed in this session
    
    processed_names.add(name)
    current_time = datetime.now()
    date_str = current_time.strftime('%Y-%m-%d')
    time_str = current_time.strftime('%H:%M:%S')
    
    if os.path.exists('attendance.csv'):
        with open('attendance.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                data = line.strip().split(',')
                if len(data) == 3:
                    recorded_name, recorded_date, recorded_time = data
                    recorded_datetime = datetime.strptime(f'{recorded_date} {recorded_time}', '%Y-%m-%d %H:%M:%S')
                    if name == recorded_name and (current_time - recorded_datetime) < timedelta(hours=24):
                        print(f'Attendance for {name} is already marked within the last 24 hours.')
                        return
    
    with open('attendance.csv', 'a') as f:
        f.write(f'{name},{date_str},{time_str}\n')
        print(f'Attendance marked for {name} on {date_str} at {time_str}')

encode_list = finEncoding(studentimg)

vid = cv2.VideoCapture(0)

while True:
    success, frame = vid.read()
    if not success:
        break
    
    smaller_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    
    faces_in_frame = face_rec.face_locations(smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(smaller_frames, faces_in_frame)
    
    for encodeFace, faceloc in zip(encodeFacesInFrame, faces_in_frame):
        matches = face_rec.compare_faces(encode_list, encodeFace)
        facedis = face_rec.face_distance(encode_list, encodeFace)
        matchIndex = np.argmin(facedis)
        
        if matches[matchIndex]:
            name = studentNames[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)
    
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
