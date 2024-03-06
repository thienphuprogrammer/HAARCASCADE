import cv2


def draw_boundary(frame, classifier, scaleFactor=1.1, minNeighbors=5, color=(0, 255, 0), text="Face"):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coordinates = []

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        coordinates = [x, y, w, h]

    return coordinates


def detect(frame, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255)}
    coordinates = draw_boundary(frame, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coordinates) == 4:
        roi_frame = frame[coordinates[1]:coordinates[1]+coordinates[3], coordinates[0]:coordinates[0]+coordinates[2]]
        coordinates = draw_boundary(roi_frame, eyeCascade, 1.1, 14, color['green'], "Eye")

    return frame


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouthCascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = detect(frame, faceCascade, eyeCascade, noseCascade, mouthCascade)
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
