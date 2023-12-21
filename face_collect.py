
import cv2
import numpy

# init camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

skip = 0
face_data = []
offset = 10
dataset_path = './face_data/'
file_name = input("enter the name of person: ")

while True:
    ret,frame = cap.read()

    if ret==False:
        continue

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    # sort face based upon area w*h
    faces = sorted(faces,key=lambda f:f[2]*f[3])

    # Pick the largest face
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        # crop out required face (region of interest)
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        cv2.imshow("face section",face_section)

        skip+=1
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))


    cv2.imshow("Frame",frame)


    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data = numpy.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))

numpy.save(dataset_path + file_name +'.npy',face_data)
print("data saved")

cap.release()
cap.destroyAllWindows()
