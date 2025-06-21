import cv2
import numpy as np

# Load Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        if len(eyes) >= 2 and len(noses) >= 1:
            # Sort eyes left to right
            eyes = sorted(eyes, key=lambda e: e[0])

            # Left Eye
            ex, ey, ew, eh = eyes[0]
            eye_img = roi_color[ey:ey+eh, ex:ex+ew].copy()

            # Nose
            nx, ny, nw, nh = noses[0]
            nose_img = roi_color[ny:ny+nh, nx:nx+nw].copy()

            # Draw rectangles
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)  # Eye box - Blue
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 255, 0), 2)  # Nose box - Green

            # Resize for swapping
            nose_resized = cv2.resize(nose_img, (ew, eh))
            eye_resized = cv2.resize(eye_img, (nw, nh))

            # Swap: nose to eye, eye to nose
            roi_color[ey:ey+eh, ex:ex+ew] = nose_resized
            roi_color[ny:ny+nh, nx:nx+nw] = eye_resized

    # Show frame
    cv2.imshow('Real-Time Eye-Nose Swap', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
