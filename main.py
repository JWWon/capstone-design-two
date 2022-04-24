import cv2
import numpy as np
import light_remover as lr
import imutils
from imutils.video import VideoStream
from keras.models import load_model
import pygame


def main():
    leye = cv2.CascadeClassifier('pre_trained/haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('pre_trained/haarcascade_righteye_2splits.xml')
    model = load_model('model/weight.h5')
    pygame.mixer.init()
    warning_sound = pygame.mixer.Sound('asset/windsheld.mp3')
    danger_sound = pygame.mixer.Sound('asset/pullup.mp3')

    video = VideoStream(src=0).start()

    # const
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    target_size = (24,24)
    activate_threshold = 15

    # variable
    status = 0 # 0: stay awake, 1: drowsy, 2: sleep
    both_eyes_closed_count = 0
    thickness = 2

    def get_eye_open(frame, bound) -> bool:
        nonlocal model
        for (x,y,w,h) in bound:
            eye = frame[y:y+h, x:x+w] 
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) # Paint eye bound box
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) # using only for reshaping 
            eye = cv2.resize(eye, target_size) 
            eye = eye/255
            eye = eye.reshape(24, 24, -1)
            eye = np.expand_dims(eye, axis=0)
            pred = model.predict_classes(eye) 
            return True if pred[0] == 1 else False
        return False

    while True:
        frame = video.read()
        frame = imutils.resize(frame, width=640)
        height, width = frame.shape[:2] 
        cv2.rectangle(frame, (0, height-50), (200, height), (255,0,0), thickness=cv2.FILLED)

        L, gray = lr.light_removing(frame)
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        left_eye_open = get_eye_open(frame, left_eye)
        right_eye_open = get_eye_open(frame, right_eye)
        both_eyes_closed = not left_eye_open and not right_eye_open

        if both_eyes_closed:
            both_eyes_closed_count += 1
        else:
            both_eyes_closed_count = max(both_eyes_closed_count - 1, 0)
        cv2.putText(frame, "Closed" if both_eyes_closed else "Open", (10, height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Count: {both_eyes_closed_count}", (100, height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
        if both_eyes_closed_count > activate_threshold:
            if status == 0:
                warning_sound.play(-1)

            if thickness < 16:
                thickness += 2
            else:
                thickness = max(thickness - 2, 2)

            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thickness) 
            status = 1
        else:
            if 0 < status:
                warning_sound.stop()
            status = 0

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()    
    video.stop()

if __name__ == '__main__':
  main()