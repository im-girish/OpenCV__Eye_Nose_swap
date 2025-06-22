import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

swap_option = 0  

print("Press '1' to swap LEFT Eye with Nose")
print("Press '2' to swap RIGHT Eye with Nose")
print("Press '0' to show Original frame")
print("Press 'q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    original_frame = frame.copy()  
    swapped_frame = frame.copy()   

    gray = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = swapped_frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        if len(eyes) >= 2 and len(noses) >= 1:
            eyes = sorted(eyes, key=lambda e: e[0])  

            
            lex, ley, lew, leh = eyes[0]
            left_eye_img = roi_color[ley:ley+leh, lex:lex+lew].copy()

            
            rex, rey, rew, reh = eyes[1]
            right_eye_img = roi_color[rey:rey+reh, rex:rex+rew].copy()

            
            nx, ny, nw, nh = noses[0]
            nose_img = roi_color[ny:ny+nh, nx:nx+nw].copy()

            
            cv2.rectangle(roi_color, (lex, ley), (lex+lew, ley+leh), (255, 0, 0), 2)  
            cv2.rectangle(roi_color, (rex, rey), (rex+rew, rey+reh), (0, 0, 255), 2)  
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 255, 0), 2)        

            if swap_option == 1:  
                nose_resized = cv2.resize(nose_img, (lew, leh))
                left_eye_resized = cv2.resize(left_eye_img, (nw, nh))

                roi_color[ley:ley+leh, lex:lex+lew] = nose_resized
                roi_color[ny:ny+nh, nx:nx+nw] = left_eye_resized

            elif swap_option == 2:  
                nose_resized = cv2.resize(nose_img, (rew, reh))
                right_eye_resized = cv2.resize(right_eye_img, (nw, nh))

                roi_color[rey:rey+reh, rex:rex+rew] = nose_resized
                roi_color[ny:ny+nh, nx:nx+nw] = right_eye_resized

    
    cv2.imshow('Original Frame', original_frame)
    cv2.imshow('Swapped Frame', swapped_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        swap_option = 1
        print("Swapping LEFT Eye with Nose")
    elif key == ord('2'):
        swap_option = 2
        print("Swapping RIGHT Eye with Nose")
    elif key == ord('0'):
        swap_option = 0
        print("Showing Original Frame Only")
    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
