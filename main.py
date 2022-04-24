import cv2
import numpy as np
import light_remover as lr
import imutils
from imutils.video import VideoStream
from keras.models import load_model


def main():
  leye = cv2.CascadeClassifier('pre_trained/haarcascade_lefteye_2splits.xml')
  reye = cv2.CascadeClassifier('pre_trained/haarcascade_righteye_2splits.xml')

  model = load_model('model/weight.h5')
  video = VideoStream(src=0).start()
  
  font = cv2.FONT_HERSHEY_COMPLEX_SMALL
  target_size = (24,24)
  
  score=0
  thicc=2

  def get_eye_open(frame, bound) -> bool:
    nonlocal model
    for (x,y,w,h) in bound:
      eye = frame[y:y+h, x:x+w] 
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) # Paint eye bound box
      eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) # (-1,-1,3) -> (-1,-1,1)
      eye = cv2.resize(eye, target_size) 
      eye = eye/255
      eye = eye.reshape(24, 24, -1)
      eye = np.expand_dims(eye, axis=0)
      pred = model.predict_classes(eye) 
      return True if pred[0] == 1 else False
    return False

  while(True):
      frame = video.read()
      frame = imutils.resize(frame, width=640)
      height, width = frame.shape[:2] 

      L, gray = lr.light_removing(frame)

      left_eye = leye.detectMultiScale(gray)
      right_eye =  reye.detectMultiScale(gray)

      cv2.rectangle(frame, (0,height-50) , (200,height) , (255,0,0) , thickness=cv2.FILLED )
      target_size = (24,24)

      left_eye_open = get_eye_open(frame, left_eye)
      right_eye_open = get_eye_open(frame, right_eye)

      if not left_eye_open and not right_eye_open:
        score += 1
        cv2.putText(frame, "Closed", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
      else:
        score -= 1
        cv2.putText(frame, "Open", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)

      if score < 0:
          #눈을 뜨고 있을 경우
          score = 0
      cv2.putText(frame,'Point:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
      if score > 15:
          #눈을 일정 시간 감고 있을 경우
          
          cv2.putText(frame, "YOU ARE SLEEPING", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          #you are sleeping 멘트 출력
          # try:
          #   sound.play() #알람 재생
          # except:  # isplaying = False
          #     pass
          if thicc < 16:
              thicc = thicc+2
          else:
              thicc = thicc-2
              if thicc < 2:
                  thicc=2
          cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
          # 15이상 됬을 경우 바탕에 빨간 영역 생성
          if(score > 100):
              # sound.stop()
              print()
      
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cv2.destroyAllWindows()    
  video.stop()

if __name__ == '__main__':
  main()