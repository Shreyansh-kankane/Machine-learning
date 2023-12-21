import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

while True: 
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False: 
        continue

    #scaling_factor = 1.3 image shrink by 30%, no.of neighbour = 5
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) 

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("video frame",frame)
    # cv2.imshow("gray video frame",gray_frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()